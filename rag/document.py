import uuid
from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    metadata: dict
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
