# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class SampleDurations:
    """Class for storing a sequence of timestamps."""

    timestamps: List[float]


@dataclass
class RevoluteJoint:
    """Class for storing a sequence of positions of a revolute joint."""

    name: str
    positions: List[float]


@dataclass
class Link:
    """Class for storing a sequence of positions and orientations of a link."""

    name: str
    positions: List[float]
    orientations: List[float]


@dataclass
class MotionData:
    """Class for the intermediate format into which different kinds of MoCap data are converted
    before retargeting. The format includes joints and/or links data associated with timestamps.
    """

    Links: List[dict] = field(default_factory=list)
    Joints: List[dict] = field(default_factory=list)
    SampleDurations: List[float] = field(default_factory=list)

    @staticmethod
    def build() -> "MotionData":
        """Build an empty MotionData."""

        return MotionData()


@dataclass
class MocapMetadata:
    """Class for the meta information about the collected MoCap data, such as the joints
    and links considered in the data collection as well as the root link of the model.
    """

    root_link: str = ""
    metadata: Dict = field(default_factory=dict)

    @staticmethod
    def build() -> "MocapMetadata":
        """Build an empty MocapMetadata."""

        return MocapMetadata()

    def add_timestamp(self) -> None:
        """Indicate that the data samples are associated with timestamps."""

        self.metadata['timestamp'] = {'type': 'TimeStamp'}

    def add_revolute_joint(self, name: str) -> None:
        """Indicate that data samples include joint positions of a specific joint."""

        self.metadata[name] = {
            'type': 'Joint',
            'position': 'True',
        }

    def add_link(self,
                 name: str,
                 position: bool = True,
                 orientation: bool = True,
                 is_root: bool = False) -> None:
        """Indicate that data samples include link positions and/or orientations of a specific link."""

        self.metadata[name] = {
            'type': 'Link',
            'position': position,
            'orientation': orientation,
        }

        # Indicate whether the link is the root link
        if is_root:
            self.root_link = name
