# MiniCourt build pipeline

from src.netGenius.components.mini_court import MiniCourt
from src.netGenius import logger

stage_name = "MINICOURT BUILT STAGE"

class Minicourt_building_pipeline():

    def __init__(self,videoframes) -> None:
        logger.info(f"stage :{stage_name}")
        self.video_frames = videoframes
    
    def run(self,player_detections,ball_detections,court_keypoints):

        try:
            logger.info('initializing minicourt')
            mini_court = MiniCourt(self.video_frames[0])

            # convert positions to mini court positions
            logger.info("converting positions to minicourt")
            player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
                player_detections,
                ball_detections,
                court_keypoints
            )
            logger.info("sucessfully converted positions to minicourt")
        except Exception as e:
            logger.exception(e)
            raise e
        
        return mini_court, player_mini_court_detections , ball_mini_court_detections
    





