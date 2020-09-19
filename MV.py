import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random

random.seed(7)
class Assignment_2():

    def __init__(self,images,loc_result,video):
        self.loc_result = loc_result
        self.images = images
        self.video = video


    def save_location(self,img_name,image_save):
        print("Sucess: Image in location",self.loc_result+img_name+ ".jpg")
        # image_save = cv2.convertScaleAbs(image_save, alpha=(255.0))
        cv2.imwrite(self.loc_result+img_name+ ".jpg", image_save)

    def read_image(self,image):
        # convert into grayscale and into float 32
        input_image = cv2.imread(image)
        # Convert into Grayscal
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return img_gray

    def extract_frames(self,filename, frames):
        result = {}
        camera = cv2.VideoCapture(filename)
        last_frame = max(frames)
        frame=0
        while camera.isOpened():
            ret,img= camera.read()
            if not ret:
                break
            if frame in frames:
                result[frame] = img
            frame += 1
            if frame>last_frame:
                break
        return result

    def get_tracks(self,filename):
        camera = cv2.VideoCapture(filename)

        # initialise features to track
        while camera.isOpened():
            ret, img = camera.read()
            if ret:
                new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                conors = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                p0 = cv2.cornerSubPix(new_img, np.float32(conors), (5, 5), (-1, -1), criteria)
                break

                # initialise tracks
        index = np.arange(len(p0))
        tracks = {}
        for i in range(len(p0)):
            tracks[index[i]] = {0: p0[i]}

        frame = 0
        while camera.isOpened():
            ret, img = camera.read()
            if not ret:
                break

            frame += 1

            old_img = new_img
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # calculate optical flow
            if len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)

                # visualise points
                for i in range(len(st)):
                    if st[i]:
                        cv2.circle(img, (p1[i, 0, 0], p1[i, 0, 1]), 2, (0, 0, 255), 2)
                        cv2.line(img, (p0[i, 0, 0], p0[i, 0, 1]), (int(p0[i, 0, 0] + (p1[i][0, 0] - p0[i, 0, 0]) * 5),
                                                                   int(p0[i, 0, 1] + (p1[i][0, 1] - p0[i, 0, 1]) * 5)),
                                 (0, 0, 255), 2)

                p0 = p1[st == 1].reshape(-1, 1, 2)
                index = index[st.flatten() == 1]

            # refresh features, if too many lost
            if len(p0) < 100:
                conors_new = cv2.goodFeaturesToTrack(new_img, 200 - len(p0), 0.3, 7)

                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                new_p0 = cv2.cornerSubPix(new_img, np.float32(conors_new), (5, 5), (-1, -1), criteria)
                for i in range(len(new_p0)):
                    if np.min(np.linalg.norm((p0 - new_p0[i]).reshape(len(p0), 2), axis=1)) > 10:
                        p0 = np.append(p0, new_p0[i].reshape(-1, 1, 2), axis=0)
                        index = np.append(index, np.max(index) + 1)

            # update tracks
            for i in range(len(p0)):
                if index[i] in tracks:
                    tracks[index[i]][frame] = p0[i]
                else:
                    tracks[index[i]] = {frame: p0[i]}

            # visualise last frames of active tracks
            for i in range(len(index)):
                for f in range(frame - 20, frame):
                    if (f in tracks[index[i]]) and (f + 1 in tracks[index[i]]):
                        cv2.line(img,
                                 (tracks[index[i]][f][0, 0], tracks[index[i]][f][0, 1]),
                                 (tracks[index[i]][f + 1][0, 0], tracks[index[i]][f + 1][0, 1]),
                                 (0, 255, 0), 1)


            # cut tracks that are too long
            # for i in tracks:
            #     if len(tracks[i])>50:
            #         keys = list(tracks[i].keys())
            #         for j in keys:
            #             if j<=max(keys)-50:
            #                 del tracks[i][j]
            # k = cv2.waitKey(1)
            # if k % 256 == 27:
            #     print("Escape hit, closing...")
            #     break
            #
            # cv2.imshow("camera", img)

        camera.release()
        cv2.destroyWindow("camera")

        return tracks, frame

    # Part 1 Task A
    def P1_TA(self,as2):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((5*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for i,image in enumerate(self.images):
            #print(image)
            img = cv2.imread(image)
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7,5),None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7,5), corners2,ret)
                as2.save_location('Part_1_Task_A_'+str(i),img)

        # Part 1 Task B
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('Matrix of Part_1_Task_B_'+str(i)+" image is :",mtx)
        return mtx

    # Part 1 Task C
    def P1_TC(self,as1):
        f1=1
        images = as1.extract_frames("Assignment_MV_02_video.mp4", [f1])
        img = images[f1].copy()
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        p0 = cv2.cornerSubPix(new_img,np.float32(corners),(5,5),(-1,-1),criteria)
        print(p0)
        for i,points in enumerate(p0):
            x,y = points[0]
            cv2.circle(img,(x,y), 5, (0,255,0), -1)
            # print(x,y,i)
        # cv2.imshow("Points",img)
        as2.save_location('Part_1_Task_C_', img)
        return img


    # Part 1 Task D
    def P1_TD(self,as2):
        tracks,frames = as2.get_tracks(self.video)
        return tracks,frames

    # Part 2 Task A
    def P2_TA(self,as2,tracks,frame):
        frame1 = 0
        frame2 = frame
        correspondences = []
        for track in tracks:
            if (frame1 in tracks[track]) and (frame2 in tracks[track]):
                x1 = [tracks[track][frame1][0, 0], tracks[track][frame1][0, 1], 1]
                x2 = [tracks[track][frame2][0, 0], tracks[track][frame2][0, 1], 1]
                correspondences.append((np.array(x1), np.array(x2)))
        # print(correspondences)
        return correspondences

    # Part 2 Task B
    def P2_TB(self, as2, correspondences):
        n = len(correspondences)
        sum_xf = 0
        sum_yf = 0
        sum_zf = 0
        sum_xl = 0
        sum_yl = 0
        sum_zl = 0
        for i,points in enumerate(correspondences):
            sum_xf += points[0][0]
            sum_yf += points[0][1]
            sum_zf += points[0][2]

            sum_xl += points[1][0]
            sum_yl += points[1][1]
            sum_zl += points[1][2]

        mue = ((sum_xf/n),(sum_yf/n),(sum_zf/n))
        mue_dash = ((sum_xl/n),(sum_yl/n),(sum_zl/n))

        # print(mue)
        # print(mue_dash)

        sigma_xf = 0
        sigma_yf = 0
        sigma_zf = 0
        sigma_xl = 0
        sigma_yl = 0
        sigma_zl = 0
        for i,points in enumerate(correspondences):
            sigma_xf += np.square(points[0][0] - mue[0])
            sigma_yf += np.square(points[0][1] - mue[1])
            sigma_zf += np.square(points[0][2] - mue[2])

            sigma_xl += np.square(points[1][0] - mue[0])
            sigma_yl += np.square(points[1][1] - mue[1])
            sigma_zl += np.square(points[1][2] - mue[2])

        sigma = (np.sqrt(sigma_xf/n),np.sqrt(sigma_yf/n),np.sqrt(sigma_zf/n))
        sigma_dash = (np.sqrt(sigma_xl / n), np.sqrt(sigma_yl / n), np.sqrt(sigma_zl / n))
        # print(sigma)
        # print(sigma_dash)

        T = np.array([[1 / sigma[0], 0, -mue[0]/sigma[0]],
                      [0, 1 / sigma[1], -mue[1]/sigma[1]],
                      [0, 0, 1]])
        T_dash = np.array([[1 / sigma_dash[0], 0, -mue_dash[0]/sigma_dash[0]],
                      [0, 1 / sigma_dash[1], -mue_dash[1]/sigma_dash[1]],
                      [0, 0, 1]])

        yi_x1_f = []
        yi_dash_x2_l = []
        for x1, x2 in correspondences:
            yi_x1_f.append(np.matmul(T, x1))
            yi_dash_x2_l.append(np.matmul(T_dash, x2))

        # print("YI",yi_x1_f)
        # print("YI_DASH",yi_dash_x2_l)
        # print(len(yi_x1_f),len(yi_dash_x2_l))
        return yi_x1_f,yi_dash_x2_l,T,T_dash

    def P2_TC_D_E_F_G(self, as2, correspondences, yi, yi_dash, T, T_dash, img):
        best_outliers = len(correspondences) + 1
        best_error = 1e100
        best_H = np.eye(3)
        for iteration in range(10000):
            # Part 2 Task C
            samples_in = set(random.sample(range(len(correspondences)), 8))
            samples_out = set(range(len(correspondences))).difference(samples_in)

            A = np.zeros((0, 9))
            # print(samples_in)
            for i in samples_in:
                Ai = np.kron(np.transpose(yi[i]), yi_dash[i])
                A = np.append(A, np.array([Ai]), axis=0)
            # Part 2 Task D
            U, S, V = np.linalg.svd(A)
            F = V[8, :].reshape(3, 3).T

            U, S, V = np.linalg.svd(F)
            F = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))
            F = np.matmul(T_dash.T, np.matmul(F, T))
            # # Part 2 Task E

            c_x_x = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]])
            count_outliers = 0
            accumulate_error = 0
            for i in samples_out:
                x1, x2 = correspondences[i]
                gi = np.matmul(np.matmul(x2.T, F), x1)

                sigma_square = np.matmul(np.matmul(np.matmul(np.matmul(x2.T, F), c_x_x), F.T), x2)
                sigma_square += np.matmul(np.matmul(np.matmul(np.matmul(x1.T, F.T), c_x_x), F), x1)

                # Part 2 Task F
                T_i = np.square(gi) / sigma_square
                if T_i > 6.635:
                    count_outliers += 1
                    # Part 2 Task H_1
                    cv2.circle(img, (int(x1[0]),int(x1[1])), 5, (255, 0, 0), -1)
                    cv2.circle(img, (int(x2[0]), int(x2[1])), 5, (255, 0, 0), -1)
                else:
                    # print(x1, x2)
                    accumulate_error += T_i
                    # Part 2 Task H_1
                    cv2.circle(img, (int(x1[0]), int(x1[1])), 5, (0, 0, 255), -1)
                    cv2.circle(img, (int(x2[0]), int(x2[1])), 5, (0, 0, 255), -1)

            if count_outliers < best_outliers:
                best_error = accumulate_error
                best_outliers = count_outliers
                best_H = F
            elif count_outliers == best_outliers:
                if accumulate_error < best_error:
                    best_error = accumulate_error
                    best_outliers = count_outliers
                    best_H = F

        as2.save_location('Part_2_Task_H_', img)

        # print(best_H)
        return best_H,F

    def calculate_epipoles(self,F):
        U, S, V = np.linalg.svd(F)
        e1 = V[2, :]

        U, S, V = np.linalg.svd(F.T)
        e2 = V[2, :]
        return e1, e2

    # Part 2 Task H_2
    def P2_TH(self, as2, bestH,frame,img):
        e1, e2 = as2.calculate_epipoles(bestH)
        print(e1 / e1[2])
        print(e2 / e2[2])

    # Part 3 Task A
    def P3_TA(self,as2,F,K):
        E = np.matmul(np.matmul(K.T,F),K)
        print("E",E)

        U, S, V = np.linalg.svd(E)
        mean_s = (S[0]+ S[1]) /2.0
        S[0] = mean_s
        S[1] = mean_s

        U,S,V = np.linalg.svd(E)
        if np.linalg.det(U) < 0:
            U[:,2] *= -1
        if np.linalg.det(V) < 0:
            V[2,:] *= -1





result_save = 'output/'
images = glob.glob('images/*.png')
video = "Assignment_MV_02_video.mp4"

if __name__ == "__main__":
#     # Saves static data that can be used easily into the functions
    as2 = Assignment_2(images,result_save,video)

#     ''' Note: I have taken as1 as an arguments because i can use read, save, show directly in the code.'''

#     # --------  PART ----- 1
    K = as2.P1_TA(as2)
    img = as2.P1_TC(as2)
    tracks,frames = as2.P1_TD(as2)

#     # --------  PART ----- 2
    correspondences = as2.P2_TA(as2,tracks,frames)
    yi,yi_dash,T,T_dash = as2.P2_TB(as2,correspondences)
    bestH,F = as2.P2_TC_D_E_F_G(as2,correspondences,yi,yi_dash,T,T_dash,img)
    as2.P2_TH(as2,bestH,frames,img)


#     # --------  PART ----- 3
    as2.P3_TA(as2,F,K)











