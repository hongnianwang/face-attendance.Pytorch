# -*- coding: utf-8 -*-

import wx
import wx.grid
import pymysql
import datetime
import os
import numpy as np
import cv2
import _thread
import threading
from PIL import Image
from mtcnn.mtcnn import MTCNN
from model.model import face_learner
from utils import prepare_facebank, getDateAndTime, SelectLogcat
from config import get_config

ID_NEW_REGISTER = 160
ID_FINISH_REGISTER = 161
ID_UPDATE_REGISTER = 162

ID_START_PUNCHCARD = 190
ID_END_PUNCARD= 191

ID_SELECT_LOGCAT = 283
ID_OPEN_LOGCAT = 284
ID_CLOSE_LOGCAT = 285
ID_CHANGE_TIME = 286
ID_MANAGE_FACE = 287
# ID_LOOK_NAME = 371

conf = get_config(False)

PASSWORD = conf.password
PATH_FACE = conf.facebank_path
PUNCARD_TIME = np.load(conf.data_path/'time.npy') #存放时间
# PASSWORD = np.load(conf.facebank_path/'password.npy') #存放时间
print(PUNCARD_TIME)
print('time loaded')
#mtcnn loaded
mtcnn = MTCNN()
print('mtcnn loaded')
#learner loaded
learner = face_learner(conf, True)
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')


class WAS(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title='学生考勤系统', size=(920, 560))

        self.initMenu()
        self.initInfoText()
        self.initGallery()
        self.initDatabase()
        self.initData()

        self.puncard_time = PUNCARD_TIME
        print('puncard_time',self.puncard_time)
        self.OnUpdateRegister()


    def initData(self):

        self.id = -1
        self.name = ''
        self.unit = ''
        self.gender = ''
        self.pic_num = 0
        self.flag_registed = False


    def initMenu(self):

        menuBar = wx.MenuBar()  # 生成菜单栏
        menu_Font = wx.Font()  # Font(faceName='consolas',pointsize=20)
        menu_Font.SetPointSize(14)
        menu_Font.SetWeight(wx.BOLD)

        registerMenu = wx.Menu()  # 生成菜单
        self.new_register = wx.MenuItem(registerMenu, ID_NEW_REGISTER, '新建录入')
        self.new_register.SetBitmap(wx.Bitmap('static/new_register.png'))
        self.new_register.SetTextColour('SLATE BLUE')
        self.new_register.SetFont(menu_Font)
        registerMenu.Append(self.new_register)

        # self.finish_register = wx.MenuItem(registerMenu, ID_FINISH_REGISTER, '完成录入')
        # self.finish_register.SetBitmap(wx.Bitmap('static/finish_register.png'))
        # self.finish_register.SetTextColour('SLATE BLUE')
        # self.finish_register.SetFont(menu_Font)
        # self.finish_register.Enable(False)
        # registerMenu.Append(self.finish_register)

        self.update_register = wx.MenuItem(registerMenu, ID_UPDATE_REGISTER, '更新录入')
        self.update_register.SetBitmap(wx.Bitmap('static/update_.png'))
        self.update_register.SetTextColour('SLATE BLUE')
        self.update_register.SetFont(menu_Font)
        registerMenu.Append(self.update_register)

        puncardMenu = wx.Menu()
        self.start_punchcard = wx.MenuItem(puncardMenu, ID_START_PUNCHCARD, '开始签到')
        self.start_punchcard.SetBitmap(wx.Bitmap('static/start_punchcard.png'))
        self.start_punchcard.SetTextColour('SLATE BLUE')
        self.start_punchcard.SetFont(menu_Font)
        puncardMenu.Append(self.start_punchcard)

        self.end_puncard = wx.MenuItem(puncardMenu, ID_END_PUNCARD, '结束签到')
        self.end_puncard.SetBitmap(wx.Bitmap('static/end_puncard.png'))
        self.end_puncard.SetTextColour('SLATE BLUE')
        self.end_puncard.SetFont(menu_Font)
        self.end_puncard.Enable(False)
        puncardMenu.Append(self.end_puncard)

        adminMenu = wx.Menu()
        self.select_logcat = wx.MenuItem(adminMenu, ID_SELECT_LOGCAT, '查询考勤')
        self.select_logcat.SetBitmap(wx.Bitmap('static/select_logcat.png'))
        self.select_logcat.SetFont(menu_Font)
        self.select_logcat.SetTextColour('SLATE BLUE')
        adminMenu.Append(self.select_logcat)

        # self.open_logcat = wx.MenuItem(adminMenu, ID_OPEN_LOGCAT, '打开日志')
        # self.open_logcat.SetBitmap(wx.Bitmap('static/open_logcat.png'))
        # self.open_logcat.SetFont(menu_Font)
        # self.open_logcat.SetTextColour('SLATE BLUE')
        # adminMenu.Append(self.open_logcat)
        #
        # self.close_logcat = wx.MenuItem(adminMenu, ID_CLOSE_LOGCAT, '关闭日志')
        # self.close_logcat.SetBitmap(wx.Bitmap('static/close_logcat.png'))
        # self.close_logcat.SetFont(menu_Font)
        # self.close_logcat.SetTextColour('SLATE BLUE')
        # self.close_logcat.Enable(False)
        # adminMenu.Append(self.close_logcat)

        self.manage_face = wx.MenuItem(adminMenu, ID_MANAGE_FACE, '删除用户')
        self.manage_face.SetBitmap(wx.Bitmap('static/start_punchcard.png'))
        self.manage_face.SetFont(menu_Font)
        self.manage_face.SetTextColour('SLATE BLUE')
        adminMenu.Append(self.manage_face)

        self.chenge_time = wx.MenuItem(adminMenu, ID_CHANGE_TIME, '修改签到时间')
        self.chenge_time.SetBitmap(wx.Bitmap('static/update_.png'))
        self.chenge_time.SetFont(menu_Font)
        self.chenge_time.SetTextColour('SLATE BLUE')
        adminMenu.Append(self.chenge_time)

        menuBar.Append(registerMenu, '&人脸录入')
        menuBar.Append(puncardMenu, '&刷脸签到')
        # menuBar.Append(logcatMenu, '&考勤日志')
        menuBar.Append(adminMenu, '&管理员功能')
        self.SetMenuBar(menuBar)


        self.Bind(wx.EVT_MENU, self.OnNewRegisterClicked, id=ID_NEW_REGISTER)
        # self.Bind(wx.EVT_MENU, self.OnFinishRegisterClicked, id=ID_FINISH_REGISTER)
        self.Bind(wx.EVT_MENU, self.OnUpdateRegisterClicked, id=ID_UPDATE_REGISTER)
        self.Bind(wx.EVT_MENU, self.OnStartPunchCardClicked, id=ID_START_PUNCHCARD)
        self.Bind(wx.EVT_MENU, self.OnEndPunchCardClicked, id=ID_END_PUNCARD)
        # self.Bind(wx.EVT_MENU, self.OnOpenLogcatClicked, id=ID_OPEN_LOGCAT)
        # self.Bind(wx.EVT_MENU, self.OnCloseLogcatClicked, id=ID_CLOSE_LOGCAT)
        self.Bind(wx.EVT_MENU, self.OnSelectLogcatClicked, id=ID_SELECT_LOGCAT)
        self.Bind(wx.EVT_MENU, self.OnManageFaceClicked, id=ID_MANAGE_FACE)
        self.Bind(wx.EVT_MENU, self.OnChangeTimeClicked, id=ID_CHANGE_TIME)

    # 注册人脸
    def register_cap(self, event):
        self.cap = cv2.VideoCapture(0)
        # cap是否初始化成功
        while self.cap.isOpened():
            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()
            # 每帧数据延时1ms，延时为0读取的是静态帧
            cv2.waitKey(1)
            if flag:
                # 检测人脸
                detect = 0
                try:
                    image = Image.fromarray(im_rd[..., ::-1])  # bgr to rgb
                    bboxes, faces = mtcnn.align_multi(image, 1, conf.min_face_size)
                    bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
                    # # 取占比最大的脸
                    biggest_face = bboxes[0]
                    # 绘制矩形框
                    cv2.rectangle(im_rd, (biggest_face[0], biggest_face[1]), (biggest_face[2], biggest_face[3]),
                                  (0, 0, 255), 3)
                    cv2.putText(im_rd, getDateAndTime(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    img_height, img_width = im_rd.shape[:2]
                    image_ = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                    pic = wx.Bitmap.FromBuffer(img_width, img_height, image_)
                    # 显示图片在panel上
                    self.bmp.SetBitmap(pic)

                    detect = 1
                except:
                    print('检测错误')

                # 识别
                if detect:
                    # try:
                    results, score = learner.infer(conf, faces, self.targets, True)
                    print('识别结果:{}({})'.format(self.knew_name, results[0]))
                    if results[0] != -1:
                        self.infoText.AppendText('[{}]人脸已注册：{}\r\n'.format(getDateAndTime(hms=1), self.knew_name[results[0]]))
                        self.flag_registed = True
                        self.OnFinishRegister()
                        _thread.exit()
                    cv2.imencode('.jpg', np.array(faces[0])[..., ::-1])[1].tofile(
                        PATH_FACE + '{}_{}_{}_{}'.format(str(self.id), self.unit, self.name, self.gender) + '/{}.jpg'.format(self.pic_num))
                    # cv2.imwrite(PATH_FACE + self.name+'/{}.jpg'.format(self.pic_num), np.array(faces[0])[..., ::-1])
                    self.pic_num += 1
                    print('写入本地：', PATH_FACE + self.name + '/{}.jpg'.format(self.pic_num))
                    self.infoText.AppendText('[{}]数据已采集：{}{}张图像\r\n'.format(getDateAndTime(hms=1), self.name, self.pic_num))
                    cv2.waitKey(5)
                    # except:
                        # print('请对准摄像头')

                if self.new_register.IsEnabled():
                    _thread.exit()
                if self.pic_num == 5:
                    self.OnFinishRegister()
                    _thread.exit()

    def OnNewRegisterClicked(self, event):
        self.new_register.Enable(False)
        # self.finish_register.Enable(True)
        self.loadDataBase(1, print_name=True)
        # 输入id，循环的逻辑是：当id=-1——>取消注册，
        #                     当id!=-1——>检测是否重号，重号——>重新输入，
        #                                            不重号——>继续下面的流程：输入name
        while self.id == -1:
            self.id = wx.GetTextFromUser(message='请输入您的学号(-1不可用)', caption='注册', parent=self.bmp, default_value='-1')
            self.id = int(self.id)
            # 取消修改
            if self.id == -1:
                # self.infoText.AppendText(
                #     '[%s]' % getDateAndTime(hms=1) + '取消添加\r\n')
                print('取消注册')
                self.new_register.Enable(True)
                # self.finish_register.Enable(False)
                break

            # 检查重号
            for knew_id in self.knew_id:
                if self.id == knew_id:
                    wx.MessageBox(message='学号已存在，请重新输入', caption='警告')
                    self.id = -1

        # 如果不重号就开始输入name，循环的逻辑是：当name=None——>取消注册，
        #                                     当name!=None——>检测是否重名，重名——>重新输入，
        #                                                                不重名——>继续下面的流程：人脸注册
        while self.id != -1 and self.unit == '':
            choice_unit = wx.SingleChoiceDialog(parent=self.bmp,
                                              message='请选择您的学院',
                                              caption='注册',
                                              choices=['计算机学院', '美术学院', '舞蹈学院', '体育学院', '音乐学院', '烹饪学院'],
                                              )
            if choice_unit.ShowModal() == wx.ID_CANCEL:
                print('取消注册')
                self.new_register.Enable(True)
                # self.finish_register.Enable(False)
                self.id = -1
                break
            else:
                self.unit = choice_unit.GetStringSelection()
                break

        while self.id != -1 and self.name == '':
            if self.id != -1:
                self.name = wx.GetTextFromUser(message='请输入您的的姓名',
                                               caption='注册',
                                               default_value='', parent=self.bmp)
                # 取消修改
                if self.name == '':
                    # self.infoText.AppendText(
                    #     '[%s]' % getDateAndTime(hms=1) + '取消添加\r\n')
                    print('取消注册')
                    # 要删除之前创建的人脸文件夹
                    self.new_register.Enable(True)
                    # self.finish_register.Enable(False)
                    self.id = -1
                    break

        while self.id != -1 and self.gender == '':
            choice_gender = wx.SingleChoiceDialog(parent=self.bmp,
                                              message='请选择您的性别',
                                              caption='注册',
                                              choices=['男', '女'],
                                              )
            self.gender = ['男', '女'][choice_gender.GetSelection()]
            if choice_gender.ShowModal() == wx.ID_CANCEL or self.gender == '':
                print('取消注册')
                self.new_register.Enable(True)
                # self.finish_register.Enable(False)
                self.id = -1
                break
                # else:
                #     # 检测是否重名
                #     for exsit_name in (os.listdir(PATH_FACE)):
                #         if exsit_name == 'facebank.pth' or exsit_name == 'names.npy':
                #             pass
                #         else:
                #             if self.name == exsit_name.split('_')[2]:
                #                 wx.MessageBox(message='姓名已存在，请重新输入', caption='警告')
                #                 self.name = ''
                #                 break
        # 没重名就开始人脸注册
        if self.id != -1 and self.name != '':
            os.makedirs(PATH_FACE + '{}_{}_{}_{}'.format(str(self.id), self.unit, self.name, self.gender), exist_ok=True)
            _thread.start_new_thread(self.register_cap, (event,))

    def OnFinishRegister(self):

        self.new_register.Enable(True)
        # self.finish_register.Enable(False)
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.pic_index))

        if self.flag_registed == True:
            dir = PATH_FACE + '{}_{}_{}_{}'.format(str(self.id), self.unit, self.name, self.gender)
            for file in os.listdir(dir):
                os.remove(dir + '/' + file)
                print('已删除已录入人脸的图片', dir + '/' + file)
            os.rmdir(dir)
            print('已删除已录入人脸的姓名文件夹', dir)
            self.initData()
            return
        if self.pic_num > 0:
            # face_descriptor = prepare_one_facebank(conf,
            #                                        learner.model,
            #                                        mtcnn,
            #                                        PATH_FACE + '{}_{}_{}_{}'.format(str(self.id), self.unit, self.name, self.gender))
            # insert to database
            self.insertRow([self.id, self.unit, self.name, self.gender], 1)
            self.infoText.AppendText('[{}]数据已保存：{}\r\n'.format(getDateAndTime(hms=1), self.name))
            self.OnUpdateRegister()
        else:
            os.rmdir(dir)
            print('已删除空文件夹', dir)

        self.initData()

    # 更新
    def OnUpdateRegister(self):
        self.initDatabase(update=1)
        self.targets, self.names = prepare_facebank(conf, learner.model, mtcnn, True)
        for i in range(len(self.targets)):
            id_, unit, name, gender = self.names[i+1].split('_')
            self.insertRow([id_, unit, name, gender], 1)
        print('人脸数据更新成功')
        self.infoText.AppendText('[%s]' % getDateAndTime(hms=1) + '人脸数据库已更新\r\n')
        self.loadDataBase(1)

    def OnUpdateRegisterClicked(self, event):
        self.OnUpdateRegister()

    # 打卡
    def punchcard_cap(self, event):
        self.cap = cv2.VideoCapture(0)
        # cap是否初始化成功
        flag_print_info = [1,1]
        flag_repeat = 0
        while self.cap.isOpened() and not self.start_punchcard.IsEnabled():
            flag, im_rd = self.cap.read()
            # 检测人脸
            try:
                image = Image.fromarray(im_rd[..., ::-1])  # bgr to rgb
                bboxes, faces = mtcnn.align_multi(image, 2, conf.min_face_size)
                bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1]  # personal choice

                results, score = learner.infer(conf, faces, self.targets, True)
                # print(results[0])
            except:
                continue
                # cv2.putText(im_rd, getDateAndTime(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                #             cv2.LINE_AA)
                # img_height, img_width = im_rd.shape[:2]
                # image_ = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                # pic = wx.Bitmap.FromBuffer(img_width, img_height, image_)
                # self.bmp.SetBitmap(pic)
                # print('请对准摄像头')
            # # 取占比最大的脸
            biggest_face = bboxes[0]
            if results[0] == -1:
                cv2.rectangle(im_rd, (biggest_face[0], biggest_face[1]), (biggest_face[2], biggest_face[3]),
                              (0, 0, 255), 2)
                cv2.putText(im_rd, 'Unknown', (biggest_face[0], biggest_face[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(im_rd, getDateAndTime(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)
                img_height, img_width = im_rd.shape[:2]
                image_ = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                pic = wx.Bitmap.FromBuffer(img_width, img_height, image_)
                # 显示图片在panel上
                self.bmp.SetBitmap(pic)
            else:
                for j, logcat_id in enumerate(self.logcat_id):
                    # 名字一样且当日日期一样，判定为签退
                    # print(logcat_id==self.knew_id[results[0]])
                    # print(getDateAndTime(ymd=True) == self.logcat_datetime_in[j][0:self.logcat_datetime_in[j].index(' ')])
                    same_id = logcat_id == self.knew_id[results[0]]
                    same_day = getDateAndTime(ymd=1) == self.logcat_datetime_in[j][0:self.logcat_datetime_in[j].index(' ')]
                    if same_id and same_day:
                        delta_seconds = (datetime.datetime.now() - datetime.datetime.strptime(self.logcat_datetime_in[j],
                                                                                '%Y-%m-%d %H:%M:%S')).seconds
                        if delta_seconds > 3*3:
                            # 间隔一段时间才能签退
                            after_check_out_time = int(getDateAndTime(hms=1).replace(':', '')) >= int(self.puncard_time[1].replace(':', ''))
                            self.updateRow([getDateAndTime(),
                                            '否' if after_check_out_time else '是',
                                            self.logcat_datetime_in[j],
                                            self.knew_name[results[0]]], 1)

                            if flag_print_info[1]:
                                self.infoText.AppendText('[{}]已签退：{}\r\n'.format(getDateAndTime(hms=1), self.knew_name[results[0]]))
                                flag_print_info[1] = 0
                        else:
                            if flag_print_info[0]:
                                self.infoText.AppendText('[{}]重复签到：{}\r\n'.format(getDateAndTime(hms=1), self.knew_name[results[0]]))
                                flag_print_info[0] = 0
                        flag_repeat = 1
                        # color = (0, 255, 0) if self.logcat_late[j] == '否' else (0, 0, 255)


                if not flag_repeat:
                    # 判断此刻的时间是否在签到时间之前
                    before_check_in_time = int(getDateAndTime(hms=1).replace(':', '')) <= int(self.puncard_time[0].replace(':', ''))
                    self.infoText.AppendText('[{}]{}: {}\r\n'.format(getDateAndTime(hms=1),
                                                                     '已签到' if before_check_in_time else '迟到了',
                                                                     self.knew_name[results[0]]))
                    self.insertRow([self.knew_id[results[0]],
                                    self.knew_unit[results[0]],
                                    self.knew_name[results[0]],
                                    getDateAndTime(),
                                    '否' if before_check_in_time else '是', '-', '-'], 2)

                    # color = (0, 255, 0) if condition else (0, 0, 255)
                    # flag_print_info = 0
                # rectangle()绘制矩形框
                # utText(): 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                cv2.rectangle(im_rd, (biggest_face[0], biggest_face[1]), (biggest_face[2], biggest_face[3]), (0, 0, 255), 2)
                cv2.putText(im_rd, 'No. %d' % self.knew_id[results[0]], (biggest_face[0], biggest_face[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(im_rd, getDateAndTime(), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                img_height, img_width = im_rd.shape[:2]
                image_ = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                pic = wx.Bitmap.FromBuffer(img_width, img_height, image_)
                # 显示图片在panel上
                self.bmp.SetBitmap(pic)
                self.loadDataBase(2)

        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.pic_index))
        _thread.exit()


    def OnStartPunchCardClicked(self, event):
        self.start_punchcard.Enable(False)
        self.end_puncard.Enable(True)
        # self.loadDataBase(1)
        self.loadDataBase(2)
        threading.Thread(target=self.punchcard_cap, args=(event,)).start()
        # _thread.start_new_thread(self.punchcard_cap,(event,))

    def OnEndPunchCardClicked(self, event):
        self.start_punchcard.Enable(True)
        self.end_puncard.Enable(False)

    # 查询
    def OnSelectLogcatClicked(self, event):
        # pass
        pw = wx.GetPasswordFromUser('请输入管理密码', caption="查询",
                               default_value="", parent=self.bmp)
        if pw == '':
            print('取消查询')
        elif pw != PASSWORD:
            wx.MessageBox(message='密码错误，退出查询', caption='警告')
        else:
            self.loadDataBase(2)
            set_logcat_id = list(set(self.logcat_id))
            set_logcat_id = [str(x) for x in set_logcat_id]
            set_logcat_unit = list(set(self.logcat_unit))

            self.logcat_datetime_in = [self.logcat_datetime_in[x][0:self.logcat_datetime_in[x].index(' ')] for x in range(len(self.logcat_datetime_in))]
            set_logcat_datetime_in = list(set(self.logcat_datetime_in))

            self.frm = SelectLogcat(set_logcat_id, set_logcat_unit, set_logcat_datetime_in, None, title='查询', size=(680, 400))
            # self.frm.SetSizer()
            self.frm.Show(True)

    def OnManageFaceClicked(self, event):
        pw = wx.GetPasswordFromUser('请输入管理密码', caption="删除",
                               default_value="", parent=self.bmp)
        if pw == '':
            print('取消删除')
        elif pw != PASSWORD:
            wx.MessageBox(message='密码错误，退出删除', caption='警告')
        else:
            self.loadDataBase(1)
            delete_id = wx.GetTextFromUser(message='请输入删除学生的学号', caption='删除', parent=self.bmp, default_value='-1')
            delete_id = int(delete_id)
            # 取消修改
            if delete_id == -1:
                # self.infoText.AppendText(
                #     '[%s]' % getDateAndTime(hms=1) + '取消添加\r\n')
                print('取消删除')
            else:
                delete_flag = 0
                for i, j in enumerate(self.knew_id):
                    if j == delete_id:
                        dlg = wx.MessageDialog(None, '请确认删除的学生为：{}学院的{}学生吗？'.format(self.knew_unit[i], self.knew_name[i]),
                                      caption='确认',
                                      style=wx.YES_NO)
                        if dlg.ShowModal() == wx.ID_YES:
                            # 删除数据库
                            self.deleteRow(i)
                        delete_flag = 1
                if not delete_flag:
                    wx.MessageBox(message='查无此人', caption='警告')





    def OnChangeTimeClicked(self, event):
        # cur_hour = datetime.datetime.now().hour
        # print(cur_hour)
        # if cur_hour>=8 or cur_hour<6:
        #     wx.MessageBox(message='''您错过了今天的签到时间，请明天再来\n
        #     每天的签到时间是:6:00~7:59''', caption='警告')
        #     return
        # pass
        pw = wx.GetPasswordFromUser('请输入管理密码', caption="修改",
                               default_value="", parent=self.bmp)
        if pw == '':
            print('取消修改')
        elif pw != PASSWORD:
            wx.MessageBox(message='密码错误，退出修改', caption='警告')
        else:
        # 输入修改时间，如果输入为空，则取消修改，如果输入格式错误，则重新修改
            flag = 1
            time_type = ['签到时间', '签退时间']
            index = -1
            while flag and index == -1:
                # 选择时间
                choice_time_type = wx.SingleChoiceDialog(parent=self.bmp,
                                                  message='请选择您要修改的时间',
                                                  caption='修改',
                                                  choices=time_type,
                                                  )
                if choice_time_type.ShowModal() == wx.ID_CANCEL:
                    print('取消修改')
                    flag = 0
                    break
                else:
                    index = choice_time_type.GetSelection()
                    # break
            # 输入时间
            # TODO 确认是否更新今日签到记录
            while flag and index != -1:
                change_time = wx.GetTextFromUser(message='请输入修改的时间(格式HH:MM:SS)',
                                                     caption='修改{}'.format(time_type[index]),
                                                     default_value=self.puncard_time[index],
                                                     parent=self.bmp)
                if change_time.strip() == '' or change_time == self.puncard_time[index]:
                    self.infoText.AppendText('[{}]取消修改(上一次为：{})\r\n'.format(getDateAndTime(hms=1), self.puncard_time[index]))
                    print('取消修改')
                    index = -1
                    break
                else:
                    try:
                        # 判断时间大小
                        changed = datetime.datetime.strptime(change_time,'%H:%M:%S')
                        original = datetime.datetime.strptime(self.puncard_time[1-index], '%H:%M:%S')
                        if changed > original if index else changed < original:
                            self.puncard_time[index] = change_time
                            np.save(conf.data_path / 'time.npy', self.puncard_time)
                            self.infoText.AppendText('[{}]{}已修改：{}\r\n'.format(getDateAndTime(hms=1), time_type[index], self.puncard_time[index]))
                            print('修改成功%s' % self.puncard_time[index])
                            index = -1
                            break
                        else:
                            print('修改错误：签到时间需早于签退时间')
                            wx.MessageBox(message='签到时间需早于签退时间，请重新输入', caption='警告')
                    except ValueError:
                        print('格式错误(格式为HH:MM:SS)')
                        wx.MessageBox(message='格式错误(格式为HH:MM:SS)', caption='警告')
                        # self.infoText.AppendText(
                        #     '[%s]' % getDateAndTime(hms=1) + '签到时间修改失败(默认为{})\r\n'.format(self.puncard_time))
                        # print('修改失败')
                        # break

    def initInfoText(self):
        # 少了这两句infoText背景颜色设置失败，莫名奇怪
        resultText = wx.StaticText(parent=self, pos=(10, 20), size=(90, 60))
        resultText.SetBackgroundColour('red')
        self.info = '\r\n' + '[%s]' % getDateAndTime(hms=1) + '初始化成功\r\n'
        # 第二个参数水平混动条
        self.infoText = wx.TextCtrl(parent=self, size=(320, 500),
                                    style=(wx.TE_MULTILINE | wx.HSCROLL | wx.TE_READONLY))
        # 前景色，也就是字体颜色
        self.infoText.SetForegroundColour('ORANGE')
        self.infoText.SetLabel(self.info)
        # API:https://www.cnblogs.com/wangjian8888/p/6028777.html
        # 没有这样的重载函数造成'par is not a key word',只好Set
        font = wx.Font()
        font.SetPointSize(12)
        font.SetWeight(wx.BOLD)
        font.SetUnderlined(True)

        self.infoText.SetFont(font)
        self.infoText.SetBackgroundColour('TURQUOISE')

    def initGallery(self):
        self.pic_index = wx.Image('static/index.png', wx.BITMAP_TYPE_ANY).Scale(640, 500)
        self.bmp = wx.StaticBitmap(parent=self, pos=(320, 0), bitmap=wx.Bitmap(self.pic_index))
        pass


    # 数据库部分
    # 初始化数据库
    def initDatabase(self, update=0):
        # conn = sqlite3.connect('facedb.db')  # 建立数据库连接
        conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接
        cur = conn.cursor()  # 得到游标对象
        # TODO type datetime

        if update:
            cur.execute('drop table if exists worker_info')
        # cur.execute("create table student(id int ,name varchar(20),class varchar(30),age varchar(10))")
        cur.execute('''create table if not exists worker_info(id bigint not null primary key, unit text not null, name text not null, gender text not null)''')
        # cur.execute('''create table if not exists worker_info(id int primary key, unit varchar(20), name varchar(20), gender varchar(20))''')

        # cur.execute('drop table if exists logcat')
        cur.execute('''create table if not exists logcat(id bigint not null, unit text not null, name text not null, datetime_in text not null, late text not null, datetime_out text not null, early text not null)''')

        cur.close()
        conn.commit()
        conn.close()

    def insertRow(self, Row, type):
        conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接
        cur = conn.cursor()  # 得到游标对象
        if type == 1:
            # Row[0] = int(Row[0])
            # print(Row[0])
            sql = '''insert into worker_info(id, unit, name, gender) values (%s,%s,%s,%s)'''
            cur.execute(sql, (Row[0], Row[1], Row[2], Row[3]))
            # cur.execute('''INSERT INTO worker_info(id, unit, name, gender) VALUES(int(Row[0]), Row[1], Row[2], Row[3]) ''')
            # (int(Row[0]), Row[1], Row[2], adapt_array(Row[3])))
            print('{}人脸数据写入成功'.format(Row[2]))

        if type == 2:
            sql = '''insert into logcat(id, unit, name, datetime_in, late, datetime_out, early) values (%s,%s,%s,%s,%s,%s,%s)'''
            cur.execute(sql, (Row[0], Row[1], Row[2], Row[3], Row[4], Row[5], Row[6]))
            print('{}签到已记录成功'.format(Row[2]))
        cur.close()
        conn.commit()
        conn.close()

    def updateRow(self, Row, type=1):
        conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接  # 建立数据库连接
        cur = conn.cursor()  # 得到游标对象
        if type == 1:
            cur.execute('''UPDATE logcat SET datetime_out='{}', early='{}' WHERE datetime_in='{}'  '''.format(Row[0], Row[1], Row[2]))
            print('{}签退已记录成功'.format(Row[3]))
        elif type == 2:
            pass
        cur.close()
        conn.commit()
        conn.close()

    def deleteRow(self, Row):
        # conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接  # 建立数据库连接
        # cur = conn.cursor()  # 得到游标对象
        #
        # cur.execute('''DELETE FROM worker_info WHERE id = %d''' % self.knew_id[Row])
        # print('{}已删除数据库的记录'.format(self.knew_name[Row]))
        #
        # cur.close()
        # conn.commit()
        # conn.close()
        dir = PATH_FACE + '{}_{}_{}_{}'.format(str(self.knew_id[Row]), self.knew_unit[Row], self.knew_name[Row], self.knew_gender[Row])
        for file in os.listdir(dir):
            os.remove(dir + '/' + file)
            print('已删除已录入人脸的图片', dir + '/' + file)
        os.rmdir(dir)
        print('已删除已录入人脸的姓名文件夹', dir)

        self.OnUpdateRegister()

    def loadDataBase(self, type, print_name=False):

        conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接  # 建立数据库连接
        cur = conn.cursor()  # 得到游标对象
        #TODO https://www.cnblogs.com/plusUltra/p/11430721.html
        if type == 1:
            # self.knew_id = []
            # self.knew_unit = []
            # self.knew_name = []
            # self.knew_gender = []
            cur.execute('select id, unit, name, gender from worker_info')
            origin = np.array(cur.fetchall())

            self.knew_id, self.knew_unit, self.knew_name, self.knew_gender = origin.T
            self.knew_id = [int(i) for i in self.knew_id]

            for i in range(len(self.knew_name)):
                id_, unit, name, gender = self.names[i+1].split('_')
                assert str(self.knew_id[i]) == id_
                assert self.knew_unit[i] == unit
                assert self.knew_name[i] == name
                assert self.knew_gender[i] == gender

            if print_name:
                print('-------------%d name in facebank-------------' % len(self.knew_name))
                for idx, i in enumerate(self.knew_name):
                    print(idx+1, i)

        if type == 2:
            cur.execute('select id, unit, name, datetime_in, late, datetime_out, early from logcat')
            # 只取今天的日期
            # today_begin = datetime.datetime.now().strftime('%Y-%m-%d')
            # today_end = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            # cur.execute('''select id, unit, name, datetime_in, late, datetime_out, early from logcat where datetime_in between '{}' and '{}' '''.format(today_begin, today_end))
            origin = np.array(cur.fetchall())

            self.logcat_id, self.logcat_unit, self.logcat_name, self.logcat_datetime_in, self.logcat_late, self.logcat_datetime_out, self.logcat_early = [], [], [], [], [], [], []

            if origin != []:
                self.logcat_id, self.logcat_unit, self.logcat_name, self.logcat_datetime_in, self.logcat_late, self.logcat_datetime_out, self.logcat_early = origin.T
                self.logcat_id = [int(i) for i in self.logcat_id]

            # self.logcat_name = np.array(self.logcat_name)


app = wx.App()
frame = WAS()
frame.Show()
app.MainLoop()
