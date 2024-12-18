from dataclasses import dataclass

@dataclass
class VideoSample:
    file_name: str
    file_id: str
    file_subset: str