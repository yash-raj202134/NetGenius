# player and ball detections pipeline

from src.netGenius.components.player_tracker import PlayerTracker
from src.netGenius.components.ball_tracker import BallTracker
from src.netGenius import logger


stage_name = "PLAYER AND BALL DETECTIONS"

class Player_and_Ball_detection_pipeline():

    def __init__(self,videoframes) -> None:
        logger.info(f"stage :{stage_name}")
        self.video_frames = videoframes


    def runTracker(self,player_model_path,ball_model_path,player_stub_path,ball_stub_path):

        # Detect players and ball
        try:
            logger.info('Reading player and ball trained model')
            player_tracker = PlayerTracker(model_path = player_model_path)
            ball_tracker = BallTracker(model_path = ball_model_path)
            logger.info("sucessfully read the models")
            
            
            logger.info("reading the stubs files (if any)")
            player_detections = player_tracker.detect_frames(self.video_frames,read_from_stub = True,stub_path = player_stub_path)
            ball_detections = ball_tracker.detect_frames(self.video_frames,read_from_stub = True,stub_path = ball_stub_path)
            logger.info('stubs files read sucessfully')
            ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
            logger.info('ball interpolation done')

        except Exception as e:
            logger.exception(e)
            raise e
        
        return player_detections
    



