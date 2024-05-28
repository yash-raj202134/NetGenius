
from src.netGenius.utils.video_utils import (read_video,save_video,draw_frame_number)
from src.netGenius.components.player_tracker import PlayerTracker
from src.netGenius.components.ball_tracker import BallTracker
from src.netGenius.components.court_line_detector import CourtLineDetector
from src.netGenius.components.mini_court import MiniCourt
from src.netGenius.components.players_stats import calculate_stats
from src.netGenius.utils.player_stats_drawer import draw_player_stats

def main():
    # Read the input video
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)


    # Detect players and ball
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    player_detections = player_tracker.detect_frames(video_frames,read_from_stub = True,stub_path = 'stubs/tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub = True,stub_path = 'stubs/tracker_stubs/ball_detections.pkl')
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


    # Courtline detector model
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    # Minicourt
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections,ball_detections,court_keypoints
    )


    player_stats_data_df = calculate_stats(video_frames,mini_court,
                                           ball_shot_frames,ball_mini_court_detections,
                                           player_mini_court_detections)
    



    # Draw output


    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)

    ## Draw court keypoints:
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    ## Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(0,255,255))

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    # Draw frame number on top left corner
    output_video_frames = draw_frame_number(output_video_frames)


    # saving the video output
    save_video(output_video_frames, "output/output_video.avi")



from src.netGenius.pipeline.stage_01_input import Inputpipeline
from src.netGenius.pipeline.stage_02_player_and_ball_detections import Player_and_Ball_detection_pipeline
from src.netGenius.pipeline.stage_03_courtline import Courtline_Detector_pipeline
from src.netGenius.pipeline.stage_04_minicourt import Minicourt_building_pipeline
from src.netGenius import logger

if __name__ =="__main__":

    # main()
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
    


    

    # testing the output:
    try:
        logger.info('drawing the output:')
        output_video_frames = player_tracker.draw_bboxes(video_frames,player_detection)
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

        # saving the video output
        save_video(output_video_frames, "testing/output_video.avi")
        logger.info('output generated')
    except Exception as e:
        logger.exception(e)
        raise e
    






    




