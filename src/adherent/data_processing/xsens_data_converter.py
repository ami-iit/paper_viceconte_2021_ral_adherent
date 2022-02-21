# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict
from dataclasses import dataclass, asdict
from adherent.data_processing import motion_data


@dataclass
class XSensDataConverter:
    """Class for converting MoCap data collected using XSens into the intermediate MoCap data format."""

    mocap_frames: List[str]
    mocap_metadata: motion_data.MocapMetadata

    @staticmethod
    def build(mocap_filename: str, mocap_metadata: motion_data.MocapMetadata) -> "XSensDataConverter":
        """Build an XSensDataConverter."""

        # Each line in the mocap file corresponds to a different frame
        with open(mocap_filename, 'r') as stream:
            mocap_frames = stream.readlines()

        return XSensDataConverter(mocap_frames=mocap_frames, mocap_metadata=mocap_metadata)

    @staticmethod
    def extract_timestamp(mocap_frame: str) -> float:
        """Extract the timestamp from the mocap data associated with a certain frame."""

        # Split the XSens message to identify data related to the links
        mocap_frame_as_list = mocap_frame.split("XsensSuit::vLink::")

        timestamp = float(mocap_frame_as_list[0].split()[1])

        return timestamp

    def extract_links_data(self, mocap_frame: str) -> Dict:
        """Extract links data from the mocap data associated with a certain frame."""

        # Discard from the XSens message data not related to the links
        mocap_frame_as_list = mocap_frame.split("XsensSuit::vLink::")[1:]
        mocap_frame_as_list[-1] = mocap_frame_as_list[-1].split("XsensSuit::vSJoint::")[0]
        mocap_frame_as_list = mocap_frame_as_list[1::2]

        links_data = {}

        for i in range(len(mocap_frame_as_list)):

            # Further cleaning of the XSens message components
            link_info = mocap_frame_as_list[i].strip('" ()').split()
            link_info[0] = link_info[0].strip('"')
            link_name = link_info[0]

            # Skip irrelevant links
            if link_name not in self.mocap_metadata.metadata.keys():
                continue

            if link_name == "Pelvis":
                # Store position and orientation for the base (Pelvis)
                links_data[link_name] = [float(n) for n in link_info[2:9]]
            else:
                # Store orientation only for the other links
                links_data[link_name] = [float(n) for n in link_info[2:6]]

        return links_data

    def clean_mocap_frames(self) -> List:
        """Clean the mocap frames collected using XSens from irrelevant information."""

        mocap_frames_cleaned = []

        for mocap_frame in self.mocap_frames:

            # Skip empty mocap frames that sometimes occur in the dataset
            if len(mocap_frame) <= 1:
                continue

            # Extract timestamp and links data
            timestamp = self.extract_timestamp(mocap_frame)
            links_data = self.extract_links_data(mocap_frame)

            # Store timestamp and links data
            mocap_frame_cleaned = {"timestamp": timestamp}
            for link_name in links_data.keys():
                mocap_frame_cleaned[link_name] = links_data[link_name]

            # Discard the mocap frames containing incomplete information that sometimes occur in the dataset
            if len(mocap_frame_cleaned.keys()) == len(self.mocap_metadata.metadata.keys()):
                mocap_frames_cleaned.append(mocap_frame_cleaned)

        return mocap_frames_cleaned

    def convert(self) -> motion_data.MotionData:
        """Convert the collected mocap data from the original to the intermediate format."""

        # Clean mocap frames from irrelevant information
        mocap_frames_cleaned = self.clean_mocap_frames()

        motiondata = motion_data.MotionData.build()

        for key, item in self.mocap_metadata.metadata.items():

            item_type = item['type']

            # Retrieve and store timestamps for the entire dataset
            if item_type == "TimeStamp":

                timestamps = [frame["timestamp"] for frame in mocap_frames_cleaned]
                motiondata.SampleDurations = timestamps

            # Retrieve and store links data for the entire dataset
            elif item_type == "Link":

                positions = [frame[key][4:] for frame in mocap_frames_cleaned if item['position']]
                quaternions = [frame[key][:4] for frame in mocap_frames_cleaned if item['orientation']]
                link = motion_data.Link(name=key, positions=positions, orientations=quaternions)
                motiondata.Links.append(asdict(link))

        return motiondata
