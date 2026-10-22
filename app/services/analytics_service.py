import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.schemas.analytics import (
    AnalyticsRunRequest,
    AnalyticsRunResponse,
    FileUploadResponse,
    ModelFile,
    ModelInventoryResponse,
    ProcessedCsvDetail,
    ProcessedCsvItem,
    ProcessedCsvListResponse,
    ProcessedVideoDetail,
    ProcessedVideoItem,
    ProcessedVideosResponse,
    UploadAndRunResponse,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


class AnalyticsService:
    def _read_csv_rows(self, csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
        """Read an events CSV file and return (columns, rows)."""
        if not csv_path.exists():
            return [], []
        with csv_path.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows: list[dict[str, Any]] = []
            for row in reader:
                parsed_row: dict[str, Any] = {}
                for k, v in row.items():
                    if v is None or v == "":
                        parsed_row[k] = None
                    else:
                        try:
                            parsed_row[k] = int(v)
                        except ValueError:
                            try:
                                parsed_row[k] = float(v)
                            except ValueError:
                                parsed_row[k] = v
                rows.append(parsed_row)
            return columns, rows

    def list_models(self) -> ModelInventoryResponse:
        paths = [
            settings.PROJECT_ROOT / "yolo26n.pt",
            settings.PROJECT_ROOT / "weights" / "yolo26n.pt",
            settings.MODEL_DIR / "volleyball_player_best.pt",
            settings.MODEL_DIR / "volleyball_player_last.pt",
            settings.MODEL_DIR / "volleyball_ball_best.pt",
            settings.PROJECT_ROOT / "runs" / "volleyball_player_detection" / "weights" / "best.pt",
            settings.PROJECT_ROOT / "runs" / "volleyball_ball_detection" / "weights" / "best.pt",
        ]
        return ModelInventoryResponse(
            models=[ModelFile(name=path.name, path=path, exists=path.exists()) for path in paths]
        )

    async def save_upload(self, file: UploadFile) -> FileUploadResponse:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = settings.UPLOAD_DIR / Path(file.filename or "upload.mp4").name

        size = 0
        with destination.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                buffer.write(chunk)

        return FileUploadResponse(filename=destination.name, path=destination, size_bytes=size)

    def list_processed_videos(self) -> ProcessedVideosResponse:
        """
        Return only videos that were fully processed — i.e. both
        ``output/{stem}_events.csv`` AND ``output/{stem}_analytics.mp4``
        exist in the output/ folder.

        NOTE: Scans output/ directly, so videos processed from any source
        (API upload, notebook, dataset folder) are all included.
        """
        output_dir = settings.OUTPUT_DIR

        if not output_dir.exists():
            return ProcessedVideosResponse(total=0, videos=[])

        videos: list[ProcessedVideoItem] = []
        idx = 1
        for csv_file in sorted(output_dir.glob("*_events.csv")):
            stem = csv_file.name[: -len("_events.csv")]
            out_video = output_dir / f"{stem}_analytics.mp4"

            if not out_video.exists():
                continue

            original_video: Path | None = None
            if settings.UPLOAD_DIR.exists():
                for ext in settings.VIDEO_EXTENSIONS:
                    candidate = settings.UPLOAD_DIR / f"{stem}{ext}"
                    if candidate.exists():
                        original_video = candidate
                        break

            stat = out_video.stat()
            size_bytes = stat.st_size
            processed_at = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()

            filename = original_video.name if original_video else f"{stem}.mp4"

            videos.append(
                ProcessedVideoItem(
                    id=idx,
                    filename=filename,
                    size_bytes=size_bytes,
                    processed_at=processed_at,
                    out_video=str(out_video),
                    out_csv=str(csv_file),
                )
            )
            idx += 1

        return ProcessedVideosResponse(total=len(videos), videos=videos)

    def get_processed_video_detail(self, video_id: int) -> ProcessedVideoDetail:
        """
        Return full detail for a single processed video by its 1-based ID,
        including jersey number detections and all events from the events CSV.
        """
        videos_resp = self.list_processed_videos()
        match = next((v for v in videos_resp.videos if v.id == video_id), None)
        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Processed video with id {video_id} not found. Available IDs: 1 to {videos_resp.total}.",
            )

        csv_path = Path(match.out_csv)
        columns, rows = self._read_csv_rows(csv_path)

        # Look for paired _jerseys.json
        stem = csv_path.name[: -len("_events.csv")]
        jerseys_file = csv_path.parent / f"{stem}_jerseys.json"
        jerseys: dict[str, str] = {}
        if jerseys_file.exists():
            try:
                with jerseys_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        jerseys = {str(k): str(v) for k, v in data.items()}
            except Exception:
                jerseys = {}

        return ProcessedVideoDetail(
            id=match.id,
            filename=match.filename,
            size_bytes=match.size_bytes,
            processed_at=match.processed_at,
            out_video=match.out_video,
            out_csv=match.out_csv,
            jerseys=jerseys,
            n_events=len(rows),
            events=rows,
        )

    def list_processed_csvs(self) -> ProcessedCsvListResponse:
        """
        Return metadata for every ``*_events.csv`` found in output/.
        Each CSV corresponds to one successfully processed video.
        """
        output_dir = settings.OUTPUT_DIR

        if not output_dir.exists():
            return ProcessedCsvListResponse(total=0, csvs=[])

        csvs: list[ProcessedCsvItem] = []
        idx = 1
        for csv_file in sorted(output_dir.glob("*_events.csv")):
            stem = csv_file.name[: -len("_events.csv")]
            stat = csv_file.stat()
            csvs.append(
                ProcessedCsvItem(
                    id=idx,
                    csv_filename=csv_file.name,
                    csv_path=str(csv_file),
                    video_filename=f"{stem}.mp4",
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                )
            )
            idx += 1

        return ProcessedCsvListResponse(total=len(csvs), csvs=csvs)

    def get_processed_csv_detail(self, csv_id: int) -> ProcessedCsvDetail:
        """
        Return full detail for a single events CSV by its 1-based ID,
        including column names, row count, and all data rows.
        """
        csvs_resp = self.list_processed_csvs()
        match = next((c for c in csvs_resp.csvs if c.id == csv_id), None)
        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Processed CSV with id {csv_id} not found. Available IDs: 1 to {csvs_resp.total}.",
            )

        csv_path = Path(match.csv_path)
        columns, rows = self._read_csv_rows(csv_path)

        return ProcessedCsvDetail(
            id=match.id,
            csv_filename=match.csv_filename,
            csv_path=match.csv_path,
            video_filename=match.video_filename,
            size_bytes=match.size_bytes,
            created_at=match.created_at,
            columns=columns,
            row_count=len(rows),
            rows=rows,
        )

    async def upload_and_run(
        self,
        file: UploadFile,
        max_seconds: float | None,
        start_seconds: float = 0.0,
        roster: str | None = None,
        max_jersey_number: int = 30,
    ) -> UploadAndRunResponse:
        """
        Save the uploaded video to uploads/ then immediately run the full
        analytics pipeline on it.

        Parameters
        ----------
        file:
            The multipart-uploaded video file.
        max_seconds:
            Duration to process from ``start_seconds``.
            ``None`` / ``0`` → process through end of video.
        start_seconds:
            Skip warm-up / training / ceremony (match-start offset).
        roster:
            Optional comma-separated valid jersey numbers for this match.
        max_jersey_number:
            Max jersey when roster is empty.
        """
        # ── 1. Persist the uploaded file ─────────────────────────────────────
        upload_meta = await self.save_upload(file)
        video_path = Path(str(upload_meta.path))

        # ── 2. Normalise max_seconds: 0 or None → full video ─────────────────
        effective_max_seconds = None if (not max_seconds or max_seconds == 0) else max_seconds

        # ── 3. Build an AnalyticsRunRequest and delegate to run() ─────────────
        request = AnalyticsRunRequest(
            video_path=str(video_path),
            max_seconds=effective_max_seconds,
            start_seconds=start_seconds or 0.0,
            split_label="upload",
            roster=roster,
            max_jersey_number=max_jersey_number,
        )
        analytics_result = self.run(request)

        # ── 4. Merge upload metadata + analytics result into one response ─────
        return UploadAndRunResponse(
            # upload metadata
            filename=upload_meta.filename,
            size_bytes=upload_meta.size_bytes,
            # analytics fields
            video=analytics_result.video,
            out_video=analytics_result.out_video,
            out_csv=analytics_result.out_csv,
            frames=analytics_result.frames,
            n_events=analytics_result.n_events,
            jerseys={str(k): str(v) for k, v in analytics_result.jerseys.items()},
            output_rows=analytics_result.output_rows,
        )

    def run(self, request: AnalyticsRunRequest) -> AnalyticsRunResponse:
        video_path = Path(request.video_path).expanduser()
        if not video_path.is_absolute():
            video_path = settings.PROJECT_ROOT / video_path
        video_path = video_path.resolve()

        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        if video_path.suffix.lower() not in settings.VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported video file: {video_path.suffix}")

        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._copy_training_weights_if_needed()

        try:
            import torch
            from volleyball_analytics.pipeline import (
                PipelineConfig,
                VolleyballAnalyticsPipeline,
                process_video,
            )
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Missing analytics dependency: {exc.name}. Install requirements.txt before running analysis.",
            ) from exc

        local_player = settings.MODEL_DIR / "volleyball_player_best.pt"
        local_ball = settings.MODEL_DIR / "volleyball_ball_best.pt"
        use_finetuned = (
            not request.pretrained_only
            and (request.force_finetuned or local_player.exists())
        )

        roster_numbers = None
        if request.roster and str(request.roster).strip():
            roster_numbers = {
                int(x.strip())
                for x in str(request.roster).split(",")
                if x.strip().isdigit()
            }

        cfg = PipelineConfig(
            project_root=settings.PROJECT_ROOT,
            player_weights=request.player_weights,
            ball_weights=request.ball_weights,
            prefer_finetuned_player=use_finetuned,
            prefer_finetuned_ball=local_ball.exists(),
            finetuned_player=local_player,
            finetuned_ball=local_ball,
            device=0 if torch.cuda.is_available() else "cpu",
            tracker="trackers/botsort_volley.yaml",
            infer_stride=1,
            player_infer_stride=2,
            ocr_interval_sec=2.0,
            ocr_min_interval_sec=0.4,
            use_motion_fallback=True,
            ocr_on_cpu=not torch.cuda.is_available(),
            crops_dir=settings.OUTPUT_DIR / "jersey_crops",
            max_jersey_number=request.max_jersey_number,
            roster_numbers=roster_numbers,
            ocr_min_bbox_area=5000,
            ocr_blur_threshold=60.0,
            ocr_brightness_min=40.0,
            ocr_brightness_max=230.0,
            ocr_skip_confirmed=True,
            jersey_vote_min=3,
            confirm_min_avg_conf=0.60,
            ocr_min_vote_conf=0.55,
            jersey_export_min_conf=request.jersey_export_min_conf,
            jersey_event_min_conf=0.50,
            jersey_dominance_max_frac=0.22,
            cluster_max_time_gap_sec=6.0,
            cluster_max_bbox_gap_px=150.0,
            cluster_inherit_unconfirmed=True,
            enforce_unique_jersey_per_frame=True,
            reject_same_player_serve_reception=True,
            max_consecutive_same_touch=1,
            max_receptions_per_rally=1,
            max_touches_per_rally=12,
            allow_pred_contacts=False,
            max_rally_sec=25.0,
            ocr_min_player_height_px=100,
            serve_to_reception_window_sec=5.0,
            serve_grace_sec=2.8,
            point_dead_ball_sec=1.0,
            max_point_after_touch_sec=1.5,
            rally_start_min_ball_frames=2,
            rally_start_window_sec=1.5,
            min_rally_ball_frames=3,
            ball_predict_frames=10,
            mid_rally_ball_gap_sec=7.0,
            inter_rally_lockout_sec=2.0,
            min_ball_conf_yolo_contact=0.18,
            min_ball_conf_pred_contact=0.28,
            rally_gap_sec=4.5,
            contact_cooldown_sec=0.80,
            contact_dist_mid_rally_px=180.0,
            attack_to_attack_cooldown_sec=0.85,
            set_same_track_cooldown_sec=0.70,
            block_same_track_cooldown_sec=0.70,
            set_chain_max_in_window=1,
            duplicate_touch_collapse_sec=0.70,
        )

        # Normalise max_seconds: 0 → full video (same as notebook's None)
        max_seconds = None if request.max_seconds == 0 else request.max_seconds
        pipeline = VolleyballAnalyticsPipeline(cfg)
        result = process_video(
            pipeline,
            video_path,
            settings.OUTPUT_DIR / f"{video_path.stem}_analytics.mp4",
            max_seconds=max_seconds,
            split_label=request.split_label,
            start_seconds=float(request.start_seconds or 0.0),
        )
        return AnalyticsRunResponse(**result)

    def _copy_training_weights_if_needed(self) -> None:
        pairs = [
            (
                settings.PROJECT_ROOT / "runs" / "volleyball_player_detection" / "weights" / "best.pt",
                settings.MODEL_DIR / "volleyball_player_best.pt",
            ),
            (
                settings.PROJECT_ROOT / "runs" / "volleyball_ball_detection" / "weights" / "best.pt",
                settings.MODEL_DIR / "volleyball_ball_best.pt",
            ),
        ]
        for source, destination in pairs:
            if not destination.exists() and source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)


analytics_service = AnalyticsService()
