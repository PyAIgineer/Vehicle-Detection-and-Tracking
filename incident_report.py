"""
incident_report.py — Incident Intelligence Report Generator
============================================================
Aggregates all detection data, camera timeline, escape routes,
and vehicle attributes into a structured intelligence report.

Report sections:
  1. Incident summary (location, time, type)
  2. Suspected vehicle attributes
  3. Camera sightings timeline
  4. Escape route analysis
  5. Predicted current location
  6. Supporting evidence (clip references)
  7. LLM-generated narrative (Groq)
"""

import os
import cv2
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from groq import Groq

import config
from reid import GlobalVehicle
from escape_router import EscapeRoute, SpatiotemporalGraph

logger = logging.getLogger("IncidentReport")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════

class IncidentReport:
    """
    Full intelligence report for one incident.
    Serialises to JSON and generates a human-readable text summary.
    """

    def __init__(
        self,
        incident_id:     str,
        incident_type:   str,           # "crash", "flee", "theft"
        trigger_camera:  str,
        trigger_time:    datetime,
        vehicle:         GlobalVehicle,
        escape_routes:   list[EscapeRoute],
        graph:           SpatiotemporalGraph,
        trigger_frame:   Optional[object] = None,  # np.ndarray
        clip_paths:      list[str] = None,
        lpr_results:     list[dict] = None,
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
        sightings = sorted(self.vehicle.sightings, key=lambda s: s.timestamp)

        last_seen_cam   = sightings[-1].camera_id  if sightings else "unknown"
        last_seen_loc   = sightings[-1].location   if sightings else "unknown"
        last_seen_time  = sightings[-1].timestamp.isoformat() if sightings else "unknown"

        # Predicted current location (extrapolate from last sighting + best route)
        predicted_loc = "Unknown"
        if self.escape_routes:
            best = self.escape_routes[0]
            predicted_loc = (
                f"In transit toward "
                f"{config.CAMERA_LOCATIONS.get(best.segments[0].to_camera, 'unknown exit')}"
                f" (ETA ~{best.segments[0].travel_time_min:.0f} min from incident)"
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
                "global_id":     self.vehicle.global_id,
                "plate":         self.vehicle.plate or "Not detected",
                "is_motorcycle": self.vehicle.is_motorcycle,
                "vehicle_type":  "Motorcycle" if self.vehicle.is_motorcycle else "Vehicle",
                "cameras_seen":  self.vehicle.camera_count,
                "movement":      self.vehicle.movement_summary,
            },

            "camera_timeline": self.graph.get_timeline(self.vehicle.global_id),

            "escape_routes": [
                {
                    "route_id":    r.route_id,
                    "probability": r.probability,
                    "description": r.description,
                    "total_km":    r.total_km,
                    "segments": [
                        {
                            "from":      s.from_camera,
                            "to":        s.to_camera,
                            "km":        s.distance_km,
                            "bearing":   s.bearing_deg,
                            "eta_min":   s.travel_time_min,
                            "scan_cams": s.cameras_on_route,
                        }
                        for s in r.segments
                    ],
                }
                for r in self.escape_routes
            ],

            "tracking": {
                "last_seen_camera":   last_seen_cam,
                "last_seen_location": last_seen_loc,
                "last_seen_time":     last_seen_time,
                "predicted_location": predicted_loc,
                "total_sightings":    len(self.vehicle.sightings),
            },

            "evidence": {
                "clips":       self.clip_paths,
                "lpr_results": self.lpr_results,
            },

            "narrative": self.narrative,
            "movement_graph": self.graph.to_dict(),
        }

    def to_text(self) -> str:
        d = self.to_dict()
        lines = [
            "═" * 62,
            f"  SENTINEL INCIDENT INTELLIGENCE REPORT",
            f"  Incident ID  : {self.incident_id}",
            f"  Type         : {self.incident_type.upper()}",
            f"  Generated    : {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "═" * 62,
            "",
            "── INCIDENT ────────────────────────────────────────────────",
            f"  Location : {d['incident']['location']}",
            f"  Camera   : {d['incident']['camera_id']}",
            f"  Time     : {self.trigger_time.strftime('%H:%M:%S')}",
            "",
            "── SUSPECTED VEHICLE ───────────────────────────────────────",
            f"  Type     : {d['vehicle']['vehicle_type']}",
            f"  Plate    : {d['vehicle']['plate']}",
            f"  Global ID: #{d['vehicle']['global_id']}",
            f"  Cameras  : {d['vehicle']['cameras_seen']} sightings",
            "",
            "── CAMERA SIGHTINGS TIMELINE ───────────────────────────────",
        ]
        for step in d["camera_timeline"]:
            gap = " [GAP-FILL]" if step["gap_fill"] else ""
            lines.append(
                f"  [{step['seq']:02d}] {step['timestamp'][11:19]}  "
                f"{step['location']:25s}  {step['plate'] or '---':12s}{gap}"
            )

        lines += [
            "",
            "── ESCAPE ROUTE PREDICTION ─────────────────────────────────",
        ]
        for route in d["escape_routes"]:
            lines.append(
                f"  Route {route['route_id']} ({route['probability']*100:.0f}%): "
                f"{route['description']}"
            )

        lines += [
            "",
            "── PREDICTED CURRENT LOCATION ──────────────────────────────",
            f"  {d['tracking']['predicted_location']}",
            "",
        ]

        if self.narrative:
            lines += [
                "── AI NARRATIVE ────────────────────────────────────────────",
                self.narrative,
                "",
            ]

        lines.append("═" * 62)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Builds and saves incident intelligence reports.
    Optionally generates a Groq LLM narrative summary.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self._groq = Groq(api_key=api_key) if api_key else None
        if not self._groq:
            logger.warning("GROQ_API_KEY not set — narrative generation disabled.")

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
        clip_paths:     list[str] = None,
        lpr_results:    list[dict] = None,
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

        # Generate Groq narrative
        if self._groq:
            report.narrative = self._generate_narrative(report)

        # Save report
        self._save(report)
        return report

    def _generate_narrative(self, report: IncidentReport) -> str:
        """Ask Groq to write a concise intelligence narrative."""
        try:
            data = report.to_dict()
            prompt = f"""You are a law enforcement intelligence analyst.
Write a concise, factual incident intelligence narrative based on this data:

Incident Type : {data['incident_type'].upper()}
Location      : {data['incident']['location']}
Time          : {data['incident']['time']}
Vehicle       : {data['vehicle']['vehicle_type']}, Plate: {data['vehicle']['plate']}
Cameras seen  : {data['vehicle']['cameras_seen']}
Timeline      : {json.dumps(data['camera_timeline'], indent=2)}
Top escape route: {data['escape_routes'][0]['description'] if data['escape_routes'] else 'Unknown'}
Predicted location: {data['tracking']['predicted_location']}

Write 3-4 sentences. Be direct, factual, present tense. No speculation beyond what data shows.
Focus on: what happened, vehicle movement, likely current position, recommended camera priority."""

            res = self._groq.chat.completions.create(
                model=config.REPORT_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            return res.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Narrative generation failed: {e}")
            return ""

    def _save(self, report: IncidentReport) -> None:
        out_dir = config.REPORT_OUTPUT_DIR

        # JSON
        json_path = out_dir / f"{report.incident_id}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Text
        txt_path = out_dir / f"{report.incident_id}.txt"
        with open(txt_path, "w") as f:
            f.write(report.to_text())

        logger.info(f"Report saved: {json_path}")
        logger.info(f"Report text:  {txt_path}")

    def save_clip(
        self,
        camera_id: str,
        frames: list,       # list of np.ndarray
        incident_id: str,
        fps: int = 10,
    ) -> str:
        """Save a video clip of the incident."""
        if not frames:
            return ""

        h, w = frames[0].shape[:2]
        filename = config.CLIPS_OUTPUT_DIR / f"{incident_id}_{camera_id}.mp4"
        writer   = cv2.VideoWriter(
            str(filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Clip saved: {filename}")
        return str(filename)
