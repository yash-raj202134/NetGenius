# player stats calculator stage

from src.netGenius.components.players_stats import calculate_stats
from src.netGenius.utils.player_stats_drawer import draw_player_stats

from src.netGenius import logger


stage_name = "PLAYER STATS CALCULATION"

class Player_stats_pipeline():

    def __init__(self,videoframes) -> None:
        logger.info(f"stage :{stage_name}")
        self.video_frames = videoframes

    
    def run(self,mini_court,ball_tracker,ball_detections,ball_mini_court_detections,
            player_mini_court_detections):
        
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
        try:
            logger.info('creating players stats dataframe')
            player_stats_data_df = calculate_stats(
                self.video_frames,
                mini_court,
                ball_shot_frames,
                ball_mini_court_detections,
                player_mini_court_detections
            )
            logger.info('sucessfully created players dataframe')

        except Exception as e:
            logger.exception(e)
            raise e
        
        return player_stats_data_df
    
        




