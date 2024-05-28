# input pipeline 
from src.netGenius import logger

from src.netGenius.utils.video_utils import (read_video,save_video,draw_frame_number)

stage_name = "INPUT"

class Inputpipeline():
    def __init__(self,dir=None) -> None:
        self.input_video_dir = dir

    def run(self):
        try:
            logger.info(f"stage :{stage_name}")
            logger.info("Reading the input video frames")
            video_frames = read_video(self.input_video_dir)
            logger.info("successfully read the video frames")
        except Exception as e:
            logger.exception(e)
            raise e
        
        return video_frames
    


