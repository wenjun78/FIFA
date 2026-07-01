"""
Video input — uploaded file or YouTube URL.

A full match is far too long to pose-track, so for URLs the caller passes a
start/end window and only that segment is downloaded (via yt-dlp's download_ranges).
"""
import os
import tempfile
import glob


def from_upload(uploaded_file):
    """Persist a Streamlit UploadedFile to a temp path and return it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def from_youtube(url):
    """
    Download a YouTube clip with yt-dlp as a SINGLE progressive/video-only stream.
    That is a plain HTTP download — no stream merging and no segment cutting — so it
    needs NO ffmpeg. Trimming to a start/end window happens later in OpenCV (see
    pose.extract_poses), which seeks on the full file. Returns the local video path.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from e

    out_dir = tempfile.mkdtemp(prefix="refvision_yt_")
    out_tmpl = os.path.join(out_dir, "clip.%(ext)s")
    opts = {
        # Prefer a progressive mp4 (audio+video already in one file) -> single download,
        # no merge, no ffmpeg. Fall back to a single video-only mp4 (pose ignores audio).
        "format": ("best[ext=mp4][acodec!=none][vcodec!=none]/"
                   "best[acodec!=none][vcodec!=none]/"
                   "bestvideo[ext=mp4]/best"),
        "outtmpl": out_tmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(
            "Couldn't download that link. Try a different or shorter clip, update yt-dlp "
            "(pip install -U yt-dlp), and confirm the URL is public and plays in a browser. "
            f"Underlying error: {e}"
        ) from e

    files = [f for f in glob.glob(os.path.join(out_dir, "clip.*")) if not f.endswith(".part")]
    if not files:
        raise RuntimeError("Download finished but no video file was produced.")
    return files[0]
