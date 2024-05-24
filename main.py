from src.netGenius.utils.video_utils import (read_video,save_video)
from src.netGenius.components.player_tracker import PlayerTracker



def main():
    # Read the input video
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)


    # Detect players
    player_tracker = PlayerTracker(model_path='models/yolov8x')
    player_detections = player_tracker.detect_frames(video_frames)



    # Draw output

    ## Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)


    save_video(output_video_frames, "output/output_video.avi")




if __name__ =="__main__":

    main()

