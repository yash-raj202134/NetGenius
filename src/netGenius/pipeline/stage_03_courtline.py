# courtline detector model pipeline

from src.netGenius.components.court_line_detector import CourtLineDetector
from src.netGenius import logger


stage_name = "COURTLINE DETECTOR"

class Courtline_Detector_pipeline():

    def __init__(self,videoframes) -> None:
        logger.info(f"stage :{stage_name}")
        self.video_frames = videoframes

    
    def run(self,court_model_path, player_tracker,player_detections):
        logger.info('reading pretrained court model path')
        court_line_detector = CourtLineDetector(court_model_path)
        logger.info('predicting court keypoints')
        court_keypoints = court_line_detector.predict(self.video_frames[0])
        logger.info('sucessfully predicted court keypoints')

        # Choose players

        player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)
        logger.info('sucessfully filtered players')

        return court_keypoints , player_detections







