# main script
from src.netGenius.pipeline.stage_01_input import Inputpipeline
from src.netGenius.pipeline.stage_02_player_and_ball_detections import Player_and_Ball_detection_pipeline
from src.netGenius.pipeline.stage_03_courtline import Courtline_Detector_pipeline
from src.netGenius.pipeline.stage_04_minicourt import Minicourt_building_pipeline
from src.netGenius.pipeline.stage_05_player_stats_calculator import Player_stats_pipeline
from src.netGenius.pipeline.stage_06_output import Draw_output

from src.netGenius import logger


if __name__ =="__main__":

    # stage 01 : Input
    inp = Inputpipeline("input_video/input_video.mp4")
    video_frames = inp.run()

    # stage 02 : player and ball detections
    player_and_ball_detections = Player_and_Ball_detection_pipeline(video_frames)
    player_detection , ball_detection, player_tracker, ball_tracker = player_and_ball_detections.runTracker(
        player_model_path='models/yolov8x',
        ball_model_path='models/last.pt',
        player_stub_path='stubs/tracker_stubs/player_detections.pkl',
        ball_stub_path='stubs/tracker_stubs/ball_detections.pkl'
        )
    
    
    # stage 03 : Courtline 
    courtline = Courtline_Detector_pipeline(video_frames)
    court_line_detector,court_keypoints, player_detection = courtline.run(
        court_model_path = 'models/keypoints_model.pth', 
        player_tracker = player_tracker,
        player_detections = player_detection
    )
    
    # stage 04 : Minicourt
    minicourt = Minicourt_building_pipeline(video_frames)
    mini_court, player_mini_court_detections, ball_mini_court_detections = minicourt.run(
        player_detections=player_detection,
        ball_detections=ball_detection,
        court_keypoints=court_keypoints,
    )

    # stage 05 : Player stats calculations
    playerstats = Player_stats_pipeline(video_frames)
    player_stats_dataframe = playerstats.run(
        mini_court=mini_court,
        ball_tracker=ball_tracker,
        ball_detections=ball_detection,
        ball_mini_court_detections=ball_mini_court_detections,
        player_mini_court_detections=player_mini_court_detections
    )

    # stage 06 : output
    out = Draw_output(video_frames,"output/output_.avi")
    result = out.run(
        player_tracker = player_tracker,
        ball_tracker = ball_tracker,
        player_detection = player_detection,
        ball_detection = ball_detection,
        court_line_detector = court_line_detector,
        court_keypoints = court_keypoints,
        mini_court = mini_court,
        player_mini_court_detections = player_mini_court_detections,
        ball_mini_court_detections = ball_mini_court_detections,
        player_stats_dataframe = player_stats_dataframe,

    )

    logger.info(f'status : {result}')






    # # testing the output:
    # try:
    #     logger.info('drawing the output:')
    #     output_video_frames = player_tracker.draw_bboxes(video_frames,player_detection)
    #     output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detection)

    #     ## Draw court keypoints:
    #     output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    #     ## Draw mini court
    #     output_video_frames = mini_court.draw_mini_court(output_video_frames)
    #     output_video_frames = mini_court.draw_points_on_mini_court(
    #         output_video_frames,
    #         player_mini_court_detections
    #     )
    #     output_video_frames = mini_court.draw_points_on_mini_court(
    #         output_video_frames,
    #         ball_mini_court_detections,
    #         color=(0,255,255)
    #     )
    #     # Draw Player Stats
    #     output_video_frames = draw_player_stats(output_video_frames,player_stats_dataframe)

    #     # saving the video output
    #     save_video(output_video_frames, "testing/output_video.avi")
    #     logger.info('output generated')
    # except Exception as e:
    #     logger.exception(e)
    #     raise e
    
 