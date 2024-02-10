import numpy as np
import numpy.typing as npt
import cv2 as cv
from typing import Tuple, List
from pathlib import Path
import math
import random

Frame_t = npt.NDArray[np.uint8]
Pixel_t = npt.NDArray[np.uint8]

class VideoEditor():
    def __init__(self) -> None:
        self.video_paths : List[str] = []
        self.all_video_frames : List[int] = []
        self.all_video_fps : List[int] = []
        self.edited_counter : int = 0 # Counter for processed videos, so even if the same video was processed 2times, it wont have the same name when saving
        self.cut_queue : List[List[int]] = []
        self.total_time : float = 0
    def get_video_seconds(self, video_index: int) -> float:
        return self.all_video_frames[video_index]/self.all_video_fps[video_index]

    def select_videos(self, start: float, end: float) -> List[int]: # Selects which videos need to be editted based on the range of start and end
        video_indexes : List[int] = []
        seconds_processed : float = 0 # Counting how many seconds of videos were processed
        if end > self.total_time:
            end = self.total_time
        for i in range(len(self.video_paths)):

            if start < (self.get_video_seconds(i) + seconds_processed) and len(video_indexes) == 0: # Selects first video
                video_indexes.append(i)

                if end < (self.get_video_seconds(i) + seconds_processed): # If the edit ends in the same video
                    return video_indexes
            elif end < (self.get_video_seconds(i) + seconds_processed) and len(video_indexes) != 0:
                video_indexes.append(i)
                return video_indexes
            elif len(video_indexes) != 0:
                video_indexes.append(i)

            seconds_processed += self.get_video_seconds(i)
        return video_indexes # Returns empty incase no videos were selected           

    def add_video(self, path: str) -> 'VideoEditor':
        video = cv.VideoCapture(path)

        if not video.isOpened():
            print("Couldn't open video.")
            exit()

        frames : int = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps : float = video.get(cv.CAP_PROP_FPS)

        self.total_time += frames/fps

        print(f"time in seconds: {frames/fps} total frames: {frames} fps: {fps}")

        self.video_paths.append(path)
        self.all_video_frames.append(frames)
        self.all_video_fps.append(fps)

        video.release()
        return self
    
    def get_grayscale_frame(self, frame: Frame_t) -> Frame_t:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)

    def change_path(self, video_index : int) -> None:
        self.video_paths[video_index] = f'{Path(self.video_paths[video_index]).stem}output{self.edited_counter}.mp4'
        self.edited_counter += 1
        return

    def init_video_for_edit(self, video_index: int, video : 'cv.VideoCapture') -> 'cv.VideoWriter': 
        width : int = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height : int = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(f'{Path(self.video_paths[video_index]).stem}output{self.edited_counter}.mp4', fourcc, self.all_video_fps[video_index], (width, height), True)
        self.change_path(video_index)
        return out

    def grayscale(self, start: float, end: float) -> 'VideoEditor':
        videos : List[int] = self.select_videos(start, end)
        if end > self.total_time:
            end = self.total_time
        if len(videos) == 0:
            print("Video does not exist in this time interval.")
            exit()
        
        if len(videos) > 1:
            for i in range(len(videos)):
                video = cv.VideoCapture(self.video_paths[videos[i]])
                out = self.init_video_for_edit(videos[i], video)
                if i != 0 and i != len(videos)-1: # If the video is not first or last to edit, change the whole thing to grayscale
                     
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break
                        out.write(self.get_grayscale_frame(frame))

                    end -= self.get_video_seconds(videos[i]) # End gets smaller cause of already editted video
                else:        
                    start_first_video_frames : float = start * self.all_video_fps[videos[0]]
                    end_last_video_frames : float = (end - (self.get_video_seconds(videos[0])-start)) * self.all_video_fps[videos[-1]]# Where to change last video to grayscale
                    framecount : int = 0 
                    if i == 0:
                        while True:

                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break

                            if framecount > start_first_video_frames:
                                out.write(self.get_grayscale_frame(frame))
                            else:
                                out.write(frame)
                    else:
                        while True:

                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break

                            if framecount < end_last_video_frames:
                                out.write(self.get_grayscale_frame(frame))
                            else:
                                out.write(frame)
        else:
            start_first_video_frames : float = start * self.all_video_fps[videos[0]]
            end_second_video_frames : float = end * self.all_video_fps[videos[0]]

            video = cv.VideoCapture(self.video_paths[videos[0]])
            out = self.init_video_for_edit(videos[0], video)

            framecount : int = 0
            while True:

                framecount += 1
                ret, frame = video.read()
                if not ret:
                    break
                if framecount > start_first_video_frames and framecount < end_second_video_frames:
                    out.write(self.get_grayscale_frame(frame))
                else:
                    out.write(frame)

        video.release()
        out.release()
        print(self.video_paths)
        return self
    
    def chromakey(self, start: float, end: float, img: str, color: Tuple[int, int, int], similarity: int) -> 'VideoEditor':
        videos = self.select_videos(start, end)
        image = cv.imread(img)
        BGR_color = (color[2], color[1], color[0])
        if len(videos) > 1:
            for i in range(len(videos)):
                video = cv.VideoCapture(self.video_paths[videos[i]])
                width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = self.all_video_fps[videos[i]]
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter(f'{Path(self.video_paths[videos[i]]).stem}outputgray.mp4', fourcc, fps, (width, height), True)

                resized_image = cv.resize(image,(width, height))

                if i != 0 and i != len(videos)-1: # Removes videos between the first and last one
                    lower_color = np.array([0,255,255])
                    upper_color = np.array([0,0,0])
                    framecount = 0 
                    while True: # TODO - make the path change in videopaths when edited
                        framecount += 1
                        ret, frame = video.read()
                        if not ret:
                            break
                        if framecount == 1:
                            for row in range(height):
                                for col in range(width):
                                    pixel = frame[row, col]
                                    difference = abs(BGR_color[0] - pixel[0]) + abs(BGR_color[1] - pixel[1]) + abs(BGR_color[2] - pixel[2])
                                    if difference < similarity:
                                        frame[row, col] = resized_image[row,col]
                                        hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                                        hsv_pixel = hsv_frame[row, col]
                                        if hsv_pixel[1] < lower_color[1] or hsv_pixel[2] < lower_color[2]:
                                            lower_color = hsv_pixel
                                            
                                        elif hsv_pixel[1] > upper_color[1] or hsv_pixel[2] > upper_color[2]:
                                            upper_color = hsv_pixel
                                            
                            lower_color = lower_color.astype('int32')
                            upper_color = upper_color.astype('int32')

                        else:
                            hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
                            print(upper_color, lower_color)
                            mask = cv.inRange(hsv_frame, lower_color, upper_color)
                            res = cv.bitwise_and(frame, frame, mask = mask) 
  
                            f = frame - res 
                            f = np.where(f == 0, resized_image, f) 
                            cv.imshow("kjdsikf", f)
                            cv.waitKey(0)
                        out.write(frame)
                    end -= self.get_video_seconds(videos[i]) # End gets smaller cause of removed video
                else:        
                    start_first_video_frames = start * self.all_video_fps[videos[0]] # TODO - make the path change in videopaths when edited
                    end_second_video_frames = (end - (self.get_video_seconds(videos[0])-start)) * self.all_video_fps[videos[-1]]# Where to cut second video in second video time
                    if i == 0:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount > start_first_video_frames:
                                out.write(frame)
                            else:
                                out.write(frame)
                    else:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount < end_second_video_frames:
                                out.write(frame)
                            else:
                                out.write(frame)
        else:
            start_first_video_frames = start * self.all_video_fps[videos[0]]
            end_second_video_frames = end * self.all_video_fps[videos[0]]
            video = cv.VideoCapture(self.video_paths[videos[0]])
            width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = self.all_video_fps[videos[0]]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(f'{Path(self.video_paths[videos[0]]).stem}output.mp4', fourcc, fps, (width, height), True)

            resized_image = cv.resize(image,(width, height))

            framecount = 0
            while True:
                framecount += 1
                ret, frame = video.read()
                if not ret:
                    break
                if framecount > start_first_video_frames and framecount < end_second_video_frames:
                    out.write(frame)
                else:
                    out.write(frame)

        video.release()
        out.release()
        return self
    
    def cut_videos(self, start: float, end: float) -> 'VideoEditor':
        videos : List[int] = self.select_videos(start, end)
        
        if end > self.total_time:
            end = self.total_time
        if len(videos) == 0:
            print("Video does not exist in this time interval.")
            exit()
        
        if len(videos) > 1:
            for i in range(len(videos)):
                video = cv.VideoCapture(self.video_paths[videos[i]])
                blackframe = np.zeros((int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),int(video.get(cv.CAP_PROP_FRAME_WIDTH)),3), dtype = np.uint8)
                out = self.init_video_for_edit(videos[i], video)
                if i != 0 and i != len(videos)-1: # If the video is not first or last to edit, change the whole thing to grayscale
                     
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break
                        frame[0,0] = [0,255,0]
                        out.write(frame)

                    end -= self.get_video_seconds(videos[i]) # End gets smaller cause of already editted video
                else:        
                    start_first_video_frames : float = start * self.all_video_fps[videos[0]]
                    end_last_video_frames : float = (end - (self.get_video_seconds(videos[0])-start)) * self.all_video_fps[videos[-1]]# Where to change last video to grayscale
                    framecount : int = 0 
                    if i == 0:
                        while True:

                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break

                            if framecount > start_first_video_frames:
                                frame = blackframe
                                out.write(frame)
                            else:
                                out.write(frame)
                    else:
                        while True:

                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break

                            if framecount < end_last_video_frames:
                                frame = blackframe
                                out.write(frame)
                            else:
                                out.write(frame)
        else:
            start_first_video_frames : float = start * self.all_video_fps[videos[0]]
            end_second_video_frames : float = end * self.all_video_fps[videos[0]]

            video = cv.VideoCapture(self.video_paths[videos[0]])
            blackframe = np.zeros((int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),int(video.get(cv.CAP_PROP_FRAME_WIDTH)),3), dtype = np.uint8)
            out = self.init_video_for_edit(videos[0], video)

            framecount : int = 0
            while True:

                framecount += 1
                ret, frame = video.read()
                if not ret:
                    break
                if framecount > start_first_video_frames and framecount < end_second_video_frames:
                    frame = blackframe
                    out.write(frame)
                else:
                    out.write(frame)

        video.release()
        out.release()
        print(self.video_paths)
        return self

    def cut(self, start: float, end: float) -> 'VideoEditor':
        self.cut_queue.append([start, end])
        return self
    
    def zoom_in(self, img : Frame_t, randomness : int) -> Frame_t:
        x : int = random.randint(0, randomness)
        y : int = random.randint(0, randomness)
        crop : Frame_t = img[x:img.shape[0], y:img.shape[1]]
        return cv.resize(crop, (img.shape[1], img.shape[0]))

    def shaky_cam(self, start: float, end: float, randomness : int = 40) -> 'VideoEditor':
        videos = self.select_videos(start, end)
        if end > self.total_time:
            end = self.total_time
        if len(videos) > 1:
            for i in range(len(videos)):
                video = cv.VideoCapture(self.video_paths[videos[i]])
                out = self.init_video_for_edit(videos[i], video)
                if i != 0 and i != len(videos)-1: # Removes videos between the first and last one 
                    framecount = 0
                    while True: # TODO - make the path change in videopaths when edited
                        framecount +=1
                        ret, frame = video.read()
                        if not ret:
                            break
                        moved_frame = self.zoom_in(frame, randomness)
                        out.write(moved_frame)
                    end -= self.get_video_seconds(videos[i]) # End gets smaller cause of removed video
                else:        
                    start_first_video_frames = start * self.all_video_fps[videos[0]] # TODO - make the path change in videopaths when edited
                    end_second_video_frames = (end - (self.get_video_seconds(videos[0])-start)) * self.all_video_fps[videos[-1]]# Where to cut second video in second video time
                    if i == 0:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount > start_first_video_frames:
                                # EDIT FRAME HERE
                                moved_frame = self.zoom_in(frame, randomness)
                                out.write(moved_frame)
                            else:
                                out.write(frame)
                    else:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount < end_second_video_frames:
                                # EDIT FRAME HERE
                                moved_frame = self.zoom_in(frame, randomness)
                                out.write(moved_frame)
                            else:
                                out.write(frame)
        else:
            start_first_video_frames = start * self.all_video_fps[videos[0]]
            end_second_video_frames = end * self.all_video_fps[videos[0]]
            video = cv.VideoCapture(self.video_paths[videos[0]])
            out = self.init_video_for_edit(videos[0], video)
            framecount = 0
            while True:
                framecount += 1
                ret, frame = video.read()
                if not ret:
                    break
                if framecount > start_first_video_frames and framecount < end_second_video_frames:
                    # EDIT FRAME HERE
                    moved_frame = self.zoom_in(frame, randomness)
                    out.write(moved_frame)
                else:
                    out.write(frame)

        video.release()
        out.release()
        return self
    
    def detect_if_png(self, path:str) -> bool:
        if Path(path).suffix == ".png":
            return True
        return False

    def image(self, start: float, end: float, img: str, pos: Tuple[float, float, float, float]) -> 'VideoEditor':
        videos = self.select_videos(start, end)

        if end > self.total_time:
            end = self.total_time

        image = cv.imread(img)

        if image is None:
            print("Couldn't load image.")
            exit()

        if len(videos) > 1:
            for i in range(len(videos)):
                video = cv.VideoCapture(self.video_paths[videos[i]])
                out = self.init_video_for_edit(videos[i], video)

                image_width = int(((100*(pos[2]-pos[0]))*width)/100)
                image_height = int(((100*(pos[3]-pos[1]))*height)/100)
                resized_image = cv.resize(image,(image_width, image_height))

                img2gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/
                if self.detect_if_png(img):
                    ret, mask = cv.threshold(img2gray, 250, 255, cv.THRESH_BINARY_INV) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/
                else:
                    ret, mask = cv.threshold(img2gray, 0, 255, cv.THRESH_BINARY) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/
                if i != 0 and i != len(videos)-1: # Removes videos between the first and last one
                    framecount = 0 
                    while True: # TODO - make the path change in videopaths when edited
                        framecount += 1
                        ret, frame = video.read()
                        if not ret:
                            break
                        roi = frame[int(((100*pos[1])*height)/100):int(((100*pos[3])*height)/100),int(((100*pos[0])*width)/100):int(((100*pos[2])*width)/100)]
                        roi[np.where(mask)] = 0
                        roi += resized_image
                        out.write(frame)
                    end -= self.get_video_seconds(videos[i]) # End gets smaller cause of removed video
                else:        
                    start_first_video_frames = start * self.all_video_fps[videos[0]] # TODO - make the path change in videopaths when edited
                    end_second_video_frames = (end - (self.get_video_seconds(videos[0])-start)) * self.all_video_fps[videos[-1]]# Where to cut second video in second video time
                    if i == 0:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount > start_first_video_frames:
                                roi = frame[int(((100*pos[1])*height)/100):int(((100*pos[3])*height)/100),int(((100*pos[0])*width)/100):int(((100*pos[2])*width)/100)]
                                roi[np.where(mask)] = 0
                                roi += resized_image
                                out.write(frame)
                            else:
                                out.write(frame)
                    else:
                        framecount = 0 
                        while True:
                            framecount += 1
                            ret, frame = video.read()
                            if not ret:
                                break
                            if framecount < end_second_video_frames:
                                roi = frame[int(((100*pos[1])*height)/100):int(((100*pos[3])*height)/100),int(((100*pos[0])*width)/100):int(((100*pos[2])*width)/100)]
                                roi[np.where(mask)] = 0
                                roi += resized_image
                                out.write(frame)
                            else:
                                out.write(frame)
        else:
            start_first_video_frames = start * self.all_video_fps[videos[0]]
            end_second_video_frames = end * self.all_video_fps[videos[0]]
            video = cv.VideoCapture(self.video_paths[videos[0]])
            out = self.init_video_for_edit(videos[0], video)
            width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            
            image_width = int(((100*(pos[2]-pos[0]))*width)/100)
            image_height = int(((100*(pos[3]-pos[1]))*height)/100)
            # TODO check if img exists
            resized_image = cv.resize(image,(image_width, image_height))

            img2gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/
            if self.detect_if_png(img):
                ret, mask = cv.threshold(img2gray, 250, 255, cv.THRESH_BINARY_INV) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/
            else:
                ret, mask = cv.threshold(img2gray, 0, 255, cv.THRESH_BINARY) # https://www.geeksforgeeks.org/add-image-to-a-live-camera-feed-using-opencv-python/


            framecount = 0
            while True:
                framecount += 1
                ret, frame = video.read()
                if not ret:
                    break
                if framecount > start_first_video_frames and framecount < end_second_video_frames:
                    roi = frame[int(((100*pos[1])*height)/100):int(((100*pos[3])*height)/100),int(((100*pos[0])*width)/100):int(((100*pos[2])*width)/100)]
                    roi[np.where(mask)] = 0
                    roi += resized_image
                    out.write(frame)
                    pass
                else:
                    out.write(frame)

        video.release()
        out.release()
        return self
    
    def render(self, path: str, width: int, height: int, framerate: float, short: bool = False) -> 'VideoEditor':
        for vid_to_cut in self.cut_queue:
            self.cut_videos(vid_to_cut[0],vid_to_cut[1])
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(path, fourcc, framerate, (width, height), True)
        for i in range(len(self.video_paths)):
            video = cv.VideoCapture(self.video_paths[i])
            old_framerate = self.all_video_fps[i]

            if framerate > old_framerate:
                original_amount_totalframes = self.all_video_frames[i]
                new_amount_totalframes = (self.all_video_frames[i]/self.all_video_fps[i])*framerate
                new_frames = new_amount_totalframes - original_amount_totalframes
                duplicate_count = int(new_frames/original_amount_totalframes) # How many times I duplicate each frame
                decimal_duplicate_count = (new_frames/original_amount_totalframes) % 1
                add_decimals = 0
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # https://stackoverflow.com/questions/55172165/how-can-i-detect-black-frames-in-opencv-with-python
                    if np.max(grayFrame) > 20:
                        resize_frame = cv.resize(frame, (width, height))
                        for _ in range(duplicate_count):
                            out.write(resize_frame)
                        if add_decimals >=1:
                            add_decimals -= 1
                            out.write(resize_frame)
                        out.write(resize_frame)
                        add_decimals += decimal_duplicate_count
                out.write(resize_frame)

            elif framerate < old_framerate:
                    
                original_amount_totalframes = self.all_video_frames[i]
                new_amount_totalframes = (self.all_video_frames[i]/self.all_video_fps[i])*framerate
                print(f"new_amount_totalframes:{new_amount_totalframes}")
                deleted_frames = original_amount_totalframes - new_amount_totalframes
                delete_count = int(original_amount_totalframes/deleted_frames) # How many times I delete frames
                decimal_delete_count = (original_amount_totalframes/deleted_frames) % 1
                print(f"delete every: {delete_count} decimal: {decimal_delete_count}")
                add_decimals = decimal_delete_count
                counter = 0
                while True:
                    if counter+1 >= delete_count:
                        counter = 0
                        add_decimals += decimal_delete_count
                        if add_decimals >=1:
                            # counter = -1
                            add_decimals -= 1

                            ret, frame = video.read()
                            if not ret:
                                break
                            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # https://stackoverflow.com/questions/55172165/how-can-i-detect-black-frames-in-opencv-with-python
                            if np.max(grayFrame) > 20:
                                resize_frame = cv.resize(frame, (width, height))
                                out.write(resize_frame)
                        ret, _ = video.read()
                    if delete_count > 1:
                        ret, frame = video.read()
                        if not ret:
                            break
                        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # https://stackoverflow.com/questions/55172165/how-can-i-detect-black-frames-in-opencv-with-python
                        if np.max(grayFrame) > 20:
                            resize_frame = cv.resize(frame, (width, height))
                            out.write(resize_frame)
                            counter += 1
            else:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # https://stackoverflow.com/questions/55172165/how-can-i-detect-black-frames-in-opencv-with-python
                    print(np.max(grayFrame))
                    if np.max(grayFrame) > 20:
                        resize_frame = cv.resize(frame, (width, height))
                        out.write(resize_frame)

        video.release()
        out.release()
        return self
    
if __name__ == "__main__":
    # Zde si můžete své řešení testovat
    VideoEditor().add_video("ukazky\karlik_raw_footage.mp4").cut(0,11).shaky_cam(11, 17).grayscale(24, 34).cut(34,45).image(52, 60, "ukazky\\cat.png", (0, 0, 0.25, 0.25)).cut(60, 124).image(120, math.inf, "ukazky\\tux.jpg", (0, 0, 0.5, 1)).shaky_cam(100, 130).render("react_long.mp4", 426, 240, 30, False)
    # VideoEditor().add_video("ukazky\karlik_raw_footage.mp4").cut(0,11).shaky_cam(11, 17).render("react_long.mp4", 426, 240, 25, False)
    # VideoEditor().add_video("ukazky\karlik_raw_footage.mp4").add_video("ukazky\low_fps.mp4").add_video("ukazky\high_fps.mp4").shaky_cam(5,40)
    # VideoEditor().add_video("ukazky\\high_fps.mp4").shaky_cam(5,10)
    # VideoEditor().add_video("ukazky\\react_short.mp4").add_video("ukazky\\video_cut.mp4").add_video("ukazky\\images.mp4").image(5, 40, "ukazky\\cat.png", (0.5,0.5,0.75,1))
    # VideoEditor().add_video("ukazky\\video_cut.mp4").add_video("ukazky\\react_short.mp4").add_video("ukazky\\images.mp4").grayscale(5, 40).render("finalvideo.mp4",1920,1080,25)
    # VideoEditor().add_video("ukazky\\aspect_ratio.mp4").render("finalvideo.mp4",1920,1080,1)
    # VideoEditor().add_video("ukazky\\high_fps.mp4").render("finalvideo.mp4",1920,1080,5)
    # VideoEditor().add_video("finalvideo.mp4").image(5, 20, "ukazky\\tux.jpg", (0.5,0.5,0.75,1))
    # VideoEditor().add_video("karlik_raw_footageoutput0output1output2output3output4output5output6output7.mp4").render("react_long.mp4", 426, 240, 30, False)