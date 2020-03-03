import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np



class Parking:

    def show_images(self, images, cmap=None):
        cols = 2
        rows = (len(images) + 1) // cols

        plt.figure(figsize=(15, 12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            cmap = 'gray' if len(image.shape) == 2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_process(self,img):
        img_copy = np.copy(img)
        white_yellow_images = self.select_rgb_white_yellow(img)
        # park.cv_show("2",white_yellow_images)
        gray_images = self.convert_gray_scale(white_yellow_images)
        edge_images = self.detect_edges(gray_images)
        # park.cv_show("3",edge_images)
        roi_images = self.select_region(edge_images)
        # park.cv_show("4",roi_images)
        # 用霍夫变换中的直线检测把停车位的直线检测出来
        list_of_lines = self.hough_lines(roi_images)
        # 接着把线画出来
        line_images = []
        line_images = self.draw_lines(img, list_of_lines)
        # park.cv_show("5",line_images)
        rect_images, rect_coords = self.identify_blocks(img_copy, list_of_lines)
        return rect_images, rect_coords

    def hough_lines(self, image):
        # 霍夫变换要求输入的图像需要是边缘检测后的结果，不能把原始图像传进去
        # minLineLengh(线的最短长度，比这个短的都被忽略)和MaxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
        # rho距离精度,theta角度精度,threshod超过设定阈值才被检测出线段
        return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

    def select_rgb_white_yellow(self, image):
        # 过滤掉背景
        # 设置两个阈值
        lower = np.uint8([120, 120, 120])
        upper = np.uint8([255, 255, 255])
        # lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
        white_mask = cv2.inRange(image, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        return masked

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_edges(self, image, low_threshold=50, high_threshold=200):
        return cv2.Canny(image, low_threshold, high_threshold)

    def select_region(self, image):
        # first, 定义图像上的几个关键点
        rows, cols = image.shape[:2]
        pt_1 = [cols * 0.05, rows * 0.90]
        pt_2 = [cols * 0.05, rows * 0.70]
        pt_3 = [cols * 0.30, rows * 0.55]
        pt_4 = [cols * 0.6, rows * 0.15]
        pt_5 = [cols * 0.90, rows * 0.15]
        pt_6 = [cols * 0.90, rows * 0.90]

        vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
        point_img = image.copy()
        # opencv中画图时要转为BGR图像，在灰度图上画图会有问题
        point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
        for point in vertices[0]:
            cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)
        self.cv_show('point_img', point_img)

        return self.filter_region(image, vertices)

    def filter_region(self, image, vertices):
        mask = np.zeros_like(image)  # 这是一个与输入图像大小相同的一个纯黑的图像
        if len(mask.shape) == 2:
            cv2.fillPoly(mask, vertices, 255)  # 在纯黑的图像当中基于顶点坐标进行填充
            self.cv_show('mask', mask)
        return cv2.bitwise_and(image, mask)  # 把其他位置过滤掉

    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # 过滤霍夫变换检测到直线
        if make_copy:
            image = np.copy(image)
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        print(" No lines detected: ", len(cleaned))
        return image

    def identify_blocks(self, image, lines, make_copy=True):
        if make_copy:
            new_image = np.copy(image)
        # Step 1: 过滤部分直线
        cleaned = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) <= 1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
                    cleaned.append((x1, y1, x2, y2))

        # Step 2: 对直线按照x1进行排序
        import operator
        list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

        # Step 3: 找到多个列，相当于每列是一排车
        clusters = {}
        dIndex = 0
        clus_dist = 10

        for i in range(len(list1) - 1):
            distance = abs(list1[i + 1][0] - list1[i][0])
            if distance <= clus_dist:  # 满足条件的都在一列中，将线按列存在字典中
                if not dIndex in clusters.keys(): clusters[dIndex] = []
                clusters[dIndex].append(list1[i])
                clusters[dIndex].append(list1[i + 1])

            else:
                dIndex += 1

        # Step 4: 得到每一列外接矩形的坐标
        rects = {}
        i = 0
        for key in clusters:
            all_list = clusters[key]
            cleaned = list(set(all_list))
            if len(cleaned) > 5:
                cleaned = sorted(cleaned, key=lambda tup: tup[1])  # tup代表cleaned中的每一个元素，这里是按y值进行排序
                avg_y1 = cleaned[0][1]
                avg_y2 = cleaned[-1][1]
                avg_x1 = 0
                avg_x2 = 0
                for tup in cleaned:
                    avg_x1 += tup[0]
                    avg_x2 += tup[2]
                avg_x1 = avg_x1 / len(cleaned)
                avg_x2 = avg_x2 / len(cleaned)
                rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
                i += 1

        print("Num Parking Lanes: ", len(rects))
        # Step 5: 把列矩形画出来
        buff = 7
        for key in rects:
            tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
            tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
            cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 3)
        return new_image, rects

    def draw_parking(self, image, rects, make_copy=True, color=[255, 0, 0], thickness=2, save=True):
        img_copy = np.copy(image)
        for key in rects:
            tup_topLeft = (int(rects[key][0]), int(rects[key][1]))
            tup_botRight = (int(rects[key][2]), int(rects[key][3]))
            cv2.rectangle(img_copy, tup_topLeft, tup_botRight, (0, 255, 0), 2)
        self.cv_show("7", img_copy)
        if make_copy:
            new_image = np.copy(image)
        gap1 = 15.4
        gap2 = 12.6105263
        spot_dict = {}  # 字典：一个车位对应一个位置
        tot_spots = 0
        # 微调
        for key in rects:
            tup = rects[key]
            x1 = int(tup[0])
            x2 = int(tup[2])
            y1 = int(tup[1])
            y2 = int(tup[3])
            cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if key == 6:
                num_splits = int(abs(tup[3] - tup[1]) / gap2)
            else:
                num_splits = int(abs(tup[3] - tup[1]) / gap1)
            for i in range(0, num_splits + 1):
                if key == 6:
                    y = int(y1 + i * gap2)
                else:
                    y = int(y1 + i * gap1)
                cv2.line(new_image, (x1, y), (x2, y), color, thickness)
            if (x2 - x1) > 40:
                # 竖直线
                x = int((x1 + x2) / 2)
                cv2.line(new_image, (x, y1), (x, y2), color, thickness)
            # 计算数量
            if (x2 - x1) < 40:
                tot_spots += num_splits
            else:
                tot_spots += 2 * num_splits

            # 字典对应好
            if (x2 - x1) < 40:
                for i in range(0, num_splits):
                    cur_len = len(spot_dict)
                    if key == 6:
                        y = int(y1 + i * gap2)
                        spot_dict[(x1, y, x2, y + gap2)] = cur_len + 1
                    else:
                        y = int(y1 + i * gap1)
                        spot_dict[(x1, y, x2, y + gap1)] = cur_len + 1
            else:
                for i in range(0, num_splits):
                    cur_len = len(spot_dict)
                    x = int((x1 + x2) / 2)
                    if key == 6:
                        y = int(y1 + i * gap2)
                        spot_dict[(x1, y, x, y + gap2)] = cur_len + 1
                        spot_dict[(x, y, x2, y + gap2)] = cur_len + 2
                    else:
                        y = int(y1 + i * gap1)
                        spot_dict[(x1, y, x, y + gap1)] = cur_len + 1
                        spot_dict[(x, y, x2, y + gap1)] = cur_len + 2

        print("total parking spaces: ", tot_spots, cur_len)
        if save:
            filename = 'with_parking.jpg'
            cv2.imwrite(filename, new_image)
        return new_image, spot_dict

    def assign_spots_map(self, image, spot_dict, make_copy=True, color=[255, 0, 0], thickness=2):
        if make_copy:
            new_image = np.copy(image)
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return new_image

    def save_images_for_cnn(self, image, spot_dict, folder_name='cnn_data'):
        for spot in spot_dict.keys():
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            # 裁剪
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (0, 0), fx=2.0, fy=2.0)
            spot_id = spot_dict[spot]

            filename = 'spot' + str(spot_id) + '.jpg'
            print(spot_img.shape, filename, (x1, x2, y1, y2))

            cv2.imwrite("./cnn_data/{0}".format(filename), spot_img)

    def predict_on_image(self, image, spot_dict, model, class_dictionary, make_copy=True, color=[0, 255, 0], alpha=0.5):
        if make_copy:
            new_image = np.copy(image)
            overlay = np.copy(image)
        self.cv_show('new_image', new_image)
        cnt_empty = 0
        all_spots = 0
        for spot in spot_dict.keys():
            all_spots += 1
            (x1, y1, x2, y2) = spot
            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
            spot_img = image[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, (48, 48))

            label = self.make_prediction(spot_img, model, class_dictionary)
            if label == 'empty':
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                cnt_empty += 1

        cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

        cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        save = True

        if save:
            filename = 'with_marking2.jpg'
            cv2.imwrite(filename, new_image)
        self.cv_show('new_image', new_image)

        return new_image

    def predict_on_video(self, video_name, final_spot_dict, model, class_dictionary, ret=True):
        cap = cv2.VideoCapture(video_name)
        count = 0
        while ret:
            ret, image = cap.read()
            count += 1
            if count == 5:
                count = 0

                new_image = np.copy(image)
                overlay = np.copy(image)
                cnt_empty = 0
                all_spots = 0
                color = [0, 255, 0]
                alpha = 0.5
                for spot in final_spot_dict.keys():
                    all_spots += 1
                    (x1, y1, x2, y2) = spot
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    spot_img = image[y1:y2, x1:x2]
                    spot_img = cv2.resize(spot_img, (48, 48))

                    label = self.make_prediction(spot_img, model, class_dictionary)
                    if label == 'empty':
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
                        cnt_empty += 1

                cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

                cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

                cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.imshow('frame', new_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()

    def make_prediction(self, image, model, class_dictionary):
        # 预处理
        img = image / 255.

        # 转换成4D tensor，因为用的tensorflow或者keras必须处理四维的
        image = np.expand_dims(img, axis=0)

        # 用训练好的模型进行训练
        class_predicted = model.predict(image)  # 得到两个概率值
        inID = np.argmax(class_predicted[0])
        label = class_dictionary[inID]
        return label

    def rect_coords_modify(self,rect_coords):
        rect_coords[0] = (96, 484.4, 130, 499.8)
        rect_coords[1] = (96, 507.8, 130, 631.1)
        rect_coords[2] = (196, 454.6, 223, 470)
        rect_coords[3] = (169, 470, 223, 624)
        rect_coords[4] = (266, 425.2, 320, 440.6)
        rect_coords[5] = (266, 454.6, 320, 624.1)
        rect_coords[6] = (356, 391.8, 411, 631.4)
        rect_coords[7] = (486, 324.4, 513, 339.8)
        rect_coords[8] = (459, 348.8, 513, 626)
        rect_coords[9] = (580, 239.4, 607, 254.8)
        rect_coords[10] = (553, 263.8, 607, 633.5)
        rect_coords[11] = (674, 144.6, 701, 160)
        rect_coords[12] = (647, 169, 701, 631)
        rect_coords[13] = (739, 80, 793, 95.4)
        rect_coords[14] = (739, 109.4, 793, 633)
        rect_coords[15] = (833, 80, 887, 95.4)
        rect_coords[16] = (833, 109.4, 887, 633)
        rect_coords[17] = (924, 93.4, 978, 632.5)
        rect_coords[18] = (1017, 163.6, 1071, 641.1)
        rect_coords[19] = (1107, 116.4, 1141, 686.2)

        return rect_coords
