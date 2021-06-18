from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("../swissdrive.mp4",900, 1800, targetname="../cutvid.mp4")
