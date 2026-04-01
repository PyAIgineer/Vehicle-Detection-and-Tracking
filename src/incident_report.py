"""
incident_report.py — Incident Intelligence Report Generator
============================================================
Aggregates detection data, camera timeline, escape routes,
and vehicle embedding attributes into a structured report.

Sections:
  1. Incident summary (location, time, type)
  2. Vehicle attributes (color, helmet, clothing, plate)
  3. Camera sightings timeline (from spatiotemporal graph)
  4. Escape route analysis
  5. Predicted next location
  6. Evidence (clip references)
  7. LLM narrative (Groq)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from groq import Groq

import config
from reid import GlobalVehicle
from escape_router import EscapeRoute
from graph import SpatiotemporalGraph

logger = logging.getLogger("IncidentReport")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════

class IncidentReport:

    def __init__(
        self,
        incident_id:    str,
        incident_type:  str,
        trigger_camera: str,
        trigger_time:   datetime,
        vehicle:        GlobalVehicle,
        escape_routes:  list[EscapeRoute],
        graph:          SpatiotemporalGraph,
        trigger_frame=None,
        clip_paths:   list[str] = None,
        lpr_results:  list[dict] = None,
    ):
        self.incident_id    = incident_id
        self.incident_type  = incident_type
        self.trigger_camera = trigger_camera
        self.trigger_time   = trigger_time
        self.vehicle        = vehicle
        self.escape_routes  = escape_routes
        self.graph          = graph
        self.trigger_frame  = trigger_frame
        self.clip_paths     = clip_paths or []
        self.lpr_results    = lpr_results or []
        self.generated_at   = datetime.now()
        self.narrative      = ""

    def to_dict(self) -> dict:
        sightings   = sorted(self.vehicle.sightings, key=lambda s: s.timestamp)
        last        = sightings[-1] if sightings else None
        emb         = self.vehicle.embedding

        predicted_loc = "Unknown"
        if self.escape_routes:
            best = self.escape_routes[0]
            predicted_loc = (
                f"Likely heading toward "
                f"{config.CAMERA_LOCATIONS.get(best.segments[0].to_camera, 'unknown')} "
                f"(~{best.segments[0].travel_time_min*60:.0f}s ETA)"
            )

        return {
            "incident_id":   self.incident_id,
            "incident_type": self.incident_type,
            "generated_at":  self.generated_at.isoformat(),

            "incident": {
                "camera_id":   self.trigger_camera,
                "location":    config.CAMERA_LOCATIONS.get(self.trigger_camera, self.trigger_camera),
                "time":        self.trigger_time.isoformat(),
                "coordinates": config.CAMERA_COORDS.get(self.trigger_camera, []),
            },

            "vehicle": {
                "global_id":      self.vehicle.global_id,
                "plate":          self.vehicle.plate or "Not detected",
                "is_motorcycle":  self.vehicle.is_motorcycle,
                "vehicle_type":   "Motorcycle" if self.vehicle.is_motorcycle else "Vehicle",
                "cameras_seen":   self.vehicle.camera_count,
                "dominant_color": emb.dominant_color if emb else "unknown",
                "helmet_present": emb.helmet_present if emb else False,
                "helmet_color":   emb.helmet_color   if emb else "unknown",
                "clothing_color": emb.clothing_color if emb else "unknown",
                "movement":       self.vehicle.movement_summary,
            },

            "camera_timeline": self.graph.get_trajectory(self.vehicle.global_id),

            "escape_routes": [
                {
                    "route_id":    r.route_id,
                    "probability": r.probability,
                    "description": r.description,
                    "total_km":    r.total_km,
                    "segments": [
                        {
                            "from":     s.from_camera,
                            "to":       s.to_camera,
                            "km":       s.distance_km,
                            "bearing":  s.bearing_deg,
                            "eta_min":  s.travel_time_min,
                            "scan_cams":s.cameras_on_route,
                        }
                        for s in r.segments
                    ],
                }
                for r in self.escape_routes
            ],

            "tracking": {
                "last_seen_camera":   last.camera_id  if last else "unknown",
                "last_seen_location": last.location   if last else "unknown",
                "last_seen_time":     last.timestamp.isoformat() if last else "unknown",
                "predicted_location": predicted_loc,
                "total_sightings":    len(self.vehicle.sightings),
            },

            "evidence": {
                "clips":       self.clip_paths,
                "lpr_results": self.lpr_results,
            },

            "narrative":      self.narrative,
            "movement_graph": self.graph.to_dict(),
        }

    def to_text(self) -> str:
        d = self.to_dict()
        lines = [
            "═" * 62,
            "  SENTINEL INCIDENT INTELLIGENCE REPORT",
            f"  Incident ID  : {self.incident_id}",
            f"  Type         : {self.incident_type.upper()}",
            f"  Generated    : {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "═" * 62, "",
            "── INCIDENT ─────────────────────────────────────────────",
            f"  Location : {d['incident']['location']}",
            f"  Time     : {self.trigger_time.strftime('%H:%M:%S')}",
            "",
            "── VEHICLE ──────────────────────────────────────────────",
            f"  Plate    : {d['vehicle']['plate']}",
            f"  Color    : {d['vehicle']['dominant_color']}",
            f"  Helmet   : {'Yes (' + d['vehicle']['helmet_color'] + ')' if d['vehicle']['helmet_present'] else 'No'}",
            f"  Clothing : {d['vehicle']['clothing_color']}",
            f"  Cameras  : {d['vehicle']['cameras_seen']} sightings",
            "",
            "── SIGHTINGS TIMELINE ───────────────────────────────────",
        ]
        for step in d["camera_timeline"]:
            gap = " [INFERRED]" if step.get("is_gap_fill") else ""
            lines.append(
                f"  [{step.get('seq', '?'):02}] {step['timestamp'][11:19]}  "
                f"{step['location']:30s}  {step['plate'] or '---':12s}{gap}"
            )
        lines += [
            "",
            "── ESCAPE ROUTE PREDICTION ──────────────────────────────",
        ]
        for r in d["escape_routes"]:
            lines.append(f"  Route {r['route_id']} ({r['probability']*100:.0f}%): {r['description']}")
        lines += [
            "",
            f"  Predicted : {d['tracking']['predicted_location']}",
            "",
        ]
        if self.narrative:
            lines += ["── AI NARRATIVE ─────────────────────────────────────────",
                      self.narrative, ""]
        lines.append("═" * 62)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

class ReportGenerator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self._groq = Groq(api_key=api_key) if api_key else None
        if not self._groq:
            logger.warning("GROQ_API_KEY not set — narrative disabled.")
        config.REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.CLIPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        incident_type:  str,
        trigger_camera: str,
        trigger_time:   datetime,
        vehicle:        GlobalVehicle,
        escape_routes:  list[EscapeRoute],
        graph:          SpatiotemporalGraph,
        trigger_frame=None,
        clip_paths:    list[str] = None,
        lpr_results:   list[dict] = None,
    ) -> IncidentReport:
        incident_id = f"INC-{trigger_time.strftime('%Y%m%d-%H%M%S')}-{trigger_camera.upper()}"
        logger.info(f"Generating report: {incident_id}")

        report = IncidentReport(
            incident_id    = incident_id,
            incident_type  = incident_type,
            trigger_camera = trigger_camera,
            trigger_time   = trigger_time,
            vehicle        = vehicle,
            escape_routes  = escape_routes,
            graph          = graph,
            trigger_frame  = trigger_frame,
            clip_paths     = clip_paths,
            lpr_results    = lpr_results,
        )
        if self._groq:
            report.narrative = self._narrative(report)
        self._save(report)
        return report

    def _narrative(self, report: IncidentReport) -> str:
        try:
            d = report.to_dict()
            emb_info = (
                f"Vehicle color: {d['vehicle']['dominant_color']}, "
                f"helmet: {'yes (' + d['vehicle']['helmet_color'] + ')' if d['vehicle']['helmet_present'] else 'no'}, "
                f"clothing: {d['vehicle']['clothing_color']}"
            )
            prompt = (
                f"You are a surveillance intelligence analyst.\n"
                f"Write a concise 3-4 sentence incident narrative:\n\n"
                f"Type     : {d['incident_type'].upper()}\n"
                f"Location : {d['incident']['location']}\n"
                f"Time     : {d['incident']['time']}\n"
                f"Vehicle  : {d['vehicle']['vehicle_type']}, Plate: {d['vehicle']['plate']}\n"
                f"Attributes: {emb_info}\n"
                f"Cameras seen: {d['vehicle']['cameras_seen']}\n"
                f"Top route: {d['escape_routes'][0]['description'] if d['escape_routes'] else 'N/A'}\n\n"
                f"Be direct, factual, present tense. No speculation beyond data."
            )
            res = self._groq.chat.completions.create(
                model=config.REPORT_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Narrative failed: {e}")
            return ""

    def _save(self, report: IncidentReport) -> None:
        out = config.REPORT_OUTPUT_DIR
        json_path = out / f"{report.incident_id}.json"
        txt_path  = out / f"{report.incident_id}.txt"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        with open(txt_path, "w") as f:
            f.write(report.to_text())
        logger.info(f"Report saved: {json_path}")

    def save_clip(
        self, camera_id: str, frames: list,
        incident_id: str, fps: int = 10,
    ) -> str:
        if not frames:
            return ""
        h, w = frames[0].shape[:2]
        filename = config.CLIPS_OUTPUT_DIR / f"{incident_id}_{camera_id}.mp4"
        writer   = cv2.VideoWriter(
            str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Clip saved: {filename}")
        return str(filename)