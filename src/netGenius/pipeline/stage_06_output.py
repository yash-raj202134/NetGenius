# final pipeline to draw the outputs 

from src.netGenius.utils.video_utils import draw_frame_number,save_video
from src.netGenius.utils.player_stats_drawer import draw_player_stats
from src.netGenius import logger

stage_name = "OUTPUT"

class Draw_output():

    def __init__(self,videoframes,output_dir) -> None:
        logger.info(f"stage :{stage_name}")
        self.video_frames = videoframes
        self.output_dir = output_dir

    
    def run(self,player_tracker,
            ball_tracker,player_detection,
            ball_detection,court_line_detector,
            court_keypoints,mini_court,
            player_mini_court_detections,
            ball_mini_court_detections,
            player_stats_dataframe
        ):
            
        # drawing the output:
        try:
            logger.info('drawing the output:')
            output_video_frames = player_tracker.draw_bboxes(self.video_frames,player_detection)
            output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detection)

            ## Draw court keypoints:
            output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

            ## Draw mini court
            output_video_frames = mini_court.draw_mini_court(output_video_frames)
            output_video_frames = mini_court.draw_points_on_mini_court(
                output_video_frames,
                player_mini_court_detections
            )
            output_video_frames = mini_court.draw_points_on_mini_court(
                output_video_frames,
                ball_mini_court_detections,
                color=(0,255,255)
            )
            # Draw Player Stats
            output_video_frames = draw_player_stats(output_video_frames,player_stats_dataframe)

            # saving the video output
            save_video(output_video_frames, self.output_dir)
            logger.info(f'output generated and saved in {self.output_dir}')
        except Exception as e:
            logger.exception(e)
            raise e

        return True
        
