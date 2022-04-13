from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import torchvision.io.video
import os
import torch

class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('avi',), transform=None, num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, stride=2):
        super(Kinetics400, self).__init__(root)

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        split = root.split('/')[-1].strip('/')
        metadata_filepath = os.path.join(root, 'kinetics_metadata_{}.pt'.format(split))
        print(metadata_filepath)
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None
        # metadata : 
        # video_pts frames index after sampling
        # video_fps sample rate per second
        self.stride = stride
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip * stride,
            step_between_clips,
            frame_rate,
            metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.transform = transform

        if not os.path.exists(metadata_filepath):
            torch.save(self.video_clips.metadata, metadata_filepath)

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    # def __getitem__(self, idx):

    #     video, audio, info, video_idx = self.video_clips.get_clip(idx)
    #     video = self.transform(video[::self.stride]) # 3, 32, 224, 224
    #     # print(video.shape)
    #     # audio = self.transform['audio'](video)
    #     label = self.samples[video_idx][1]
    #     return video, label
    
    def __getitem__(self, idx):
        # return (video_q, video_k)
        videos = ()
        for i in idx:
            video, audio, info, video_idx = self.video_clips.get_clip(i)
            if self.transform is not None:
                video = self.transform['video'](video[::self.stride])
                videos += (video,)
        # label = self.samples[self.indices[video_idx]][1]
        return videos







