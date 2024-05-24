
from src.netGenius.utils.video_utils import (read_video,save_video)
from src.netGenius.components.player_tracker import PlayerTracker
from src.netGenius.components.ball_tracker import BallTracker



def main():
    # Read the input video
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)


    # Detect players and ball
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    player_detections = player_tracker.detect_frames(video_frames,read_from_stub = True,stub_path = 'stubs/tracker_stubs/player_detections.pkl')
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub = True,stub_path = 'stubs/tracker_stubs/ball_detections.pkl')




    # Draw output

    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames,ball_detections)


    save_video(output_video_frames, "output/output_video.avi")




if __name__ =="__main__":

    main()

