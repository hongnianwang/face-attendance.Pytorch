from PIL import Image
import numpy as np
import io
import os
from torchvision import transforms as trans
import torch
import pymysql
import datetime
import wx
import wx.grid as gridlib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

facedb = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接

# def prepare_one_facebank(conf, model, mtcnn, path, tta=True):
#     model.eval()
#     embs = []
#     for file in Path(path).iterdir():
#         if not file.is_file():
#             continue
#         else:
#             try:
#                 img = Image.open(file)
#                 print("正在处理的人脸图像：", file)
#             except:
#                 continue
#             if img.size != (112, 112):
#                 img = mtcnn.align(img)
#             with torch.no_grad():
#                 if tta:
#                     mirror = trans.functional.hflip(img)
#                     emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
#                     emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
#                     embs.append(l2_norm(emb + emb_mirror))
#                 else:
#                     embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
#
#     embedding = torch.cat(embs).mean(0, keepdim=True)
#     return embedding

def prepare_facebank(conf, model, mtcnn, tta = True):
    model.eval()
    embeddings = []
    names = ['Unknown']
    facebank_list = os.listdir(conf.facebank_path)
    for path in sorted(facebank_list, key=lambda m: int(m.split('_')[0])):
        embs = []
        for file in os.listdir(conf.facebank_path / path):
            try:
                img = Image.open(conf.facebank_path / path / file)
            except:
                print('读入失败 %s' % conf.facebank_path / path / file)
            if img.size != (112, 112):
                img = mtcnn.align(img)
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
                else:
                    embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.data_path/'facebank.pth')
    np.save(conf.data_path/'names', names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load(conf.data_path/'facebank.pth')
    names = np.load(conf.data_path/'names.npy')
    return embeddings, names

def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:            
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []
            
        results = learner.infer(conf, faces, targets, tta)
        
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice            
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0 
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1 
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

# def get_time():
#     return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
#

def getDateAndTime(ymd=False, hms=False):
    if ymd:
        # return strftime('%Y-%m-%d', localtime())
        return datetime.datetime.now().strftime('%Y-%m-%d')
    elif hms:
        # return strftime('%H:%M:%S', localtime())
        return datetime.datetime.now().strftime('%H:%M:%S')
    else:
        # return strftime('%Y-%m-%d %H:%M:%S',localtime())
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

# 查询框
class SelectLogcat(wx.Frame):
    def __init__(self, set_logcat_id, set_logcat_unit, set_logcat_datetime_in, *args, **kw):
        # 调用父类的创建方法
        super(SelectLogcat, self).__init__(*args, **kw)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer2.SetMinSize(wx.Size(1, 1))
        self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, '按学院查询', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText1.Wrap(-1)

        bSizer2.Add(self.m_staticText1, 0, wx.ALL, 8)

        m_comboBox3Choices = [''] + set_logcat_unit
        self.m_comboBox3 = wx.ComboBox(self, wx.ID_ANY, '请选择学院', wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox3Choices, 0)
        bSizer2.Add(self.m_comboBox3, 0, wx.ALL, 5)

        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, '按日期查询', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText2.Wrap(-1)

        bSizer2.Add(self.m_staticText2, 0, wx.ALL, 8)

        m_comboBox4Choices = [''] + set_logcat_datetime_in
        self.m_comboBox4 = wx.ComboBox(self, wx.ID_ANY, '请选择日期', wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox4Choices, 0)
        bSizer2.Add(self.m_comboBox4, 0, wx.ALL, 5)

        self.m_staticText3 = wx.StaticText(self, wx.ID_ANY, '按学号查询', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText3.Wrap(-1)

        bSizer2.Add(self.m_staticText3, 0, wx.ALL, 8)

        m_comboBox5Choices = [''] + set_logcat_id
        self.m_comboBox5 = wx.ComboBox(self, wx.ID_ANY, '请选择学号', wx.DefaultPosition, wx.DefaultSize,
                                       m_comboBox5Choices, 0)
        bSizer2.Add(self.m_comboBox5, 0, wx.ALL, 5)

        bSizer1.Add(bSizer2, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)

        self.m_grid2 = wx.grid.Grid(self, wx.ID_ANY, wx.Point(0, 50), wx.Size(680, 280), 0)

        # Grid
        #
        logcat_column = ['学号', '学院', '姓名', '签到时间', '迟到', '签退时间', '早退']
        self.dic = {'id': '', 'unit': '', 'date': ''}
        # self.logcat_id, self.logcat_unit, self.logcat_name, self.logcat_datetime_in, self.logcat_late, self.logcat_datetime_out, self.logcat_early = [], [], [], [], [], [], []
        self.m_grid2.CreateGrid(50, len(logcat_column))

        for i in range(50):
            for j in range(len(logcat_column)):
                self.m_grid2.SetCellAlignment(i, j, wx.ALIGN_CENTER, wx.ALIGN_CENTER)
        for i, j in enumerate(logcat_column):
            self.m_grid2.SetColLabelValue(i, j)
            if '迟' in j or '早' in j:
                self.m_grid2.SetColSize(i, 40)
            elif '时间' in j:
                self.m_grid2.SetColSize(i, 120)
            else:
                self.m_grid2.SetColSize(i, 75)
        # 初始化
        self.selectRow()
        for i, id in enumerate(self.logcat_id):
            self.m_grid2.SetCellValue(i, 0, str(id))
            self.m_grid2.SetCellValue(i, 1, self.logcat_unit[i])
            self.m_grid2.SetCellValue(i, 2, self.logcat_name[i])
            self.m_grid2.SetCellValue(i, 3, self.logcat_datetime_in[i])
            self.m_grid2.SetCellValue(i, 4, self.logcat_late[i])
            self.m_grid2.SetCellValue(i, 5, self.logcat_datetime_out[i])
            self.m_grid2.SetCellValue(i, 6, self.logcat_early[i])

        # Connect Events
        self.m_comboBox3.Bind(wx.EVT_COMBOBOX, self.OnSelect1)
        self.m_comboBox4.Bind(wx.EVT_COMBOBOX, self.OnSelect2)
        self.m_comboBox5.Bind(wx.EVT_COMBOBOX, self.OnSelect3)

        # Cell Defaults
        self.m_grid2.SetDefaultCellAlignment(wx.ALIGN_LEFT, wx.ALIGN_TOP)
        bSizer1.Add(self.m_grid2, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        bSizer3 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer3.SetMinSize(wx.Size(300, 300))
        self.m_button2 = wx.Button(self, wx.ID_ANY, '按学院统计出勤率', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button2.SetMinSize(wx.Size(150, -1))

        bSizer3.Add(self.m_button2, 0, wx.ALL, 5)

        self.m_button3 = wx.Button(self, wx.ID_ANY, '按日期统计出勤率', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button3.SetMinSize(wx.Size(150, -1))

        bSizer3.Add(self.m_button3, 0, wx.ALL, 5)

        self.m_button4 = wx.Button(self, wx.ID_ANY, '按个人统计出勤率', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button4.SetMinSize(wx.Size(150, -1))

        bSizer3.Add(self.m_button4, 0, wx.ALL, 5)

        self.m_button5 = wx.Button(self, wx.ID_ANY, '导出当前表格为csv文件', wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_button5.SetMinSize(wx.Size(150, -1))

        bSizer3.Add(self.m_button5, 0, wx.ALL, 5)

        # Connect Events
        self.m_button2.Bind(wx.EVT_BUTTON, self.OnStatistic1)
        self.m_button3.Bind(wx.EVT_BUTTON, self.OnStatistic2)
        self.m_button4.Bind(wx.EVT_BUTTON, self.OnStatistic3)
        self.m_button5.Bind(wx.EVT_BUTTON, self.OnSave)


        bSizer1.Add(bSizer3, 0, wx.ALIGN_CENTER, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)



    def OnSelect1(self, e):
        s = e.GetString()
        self.dic['unit'] = s
        print(self.dic)
        self.selectRow()
        self.m_grid2.ClearGrid()
        for i, id in enumerate(self.logcat_id):
            self.m_grid2.SetCellValue(i, 0, str(id))
            self.m_grid2.SetCellValue(i, 1, self.logcat_unit[i])
            self.m_grid2.SetCellValue(i, 2, self.logcat_name[i])
            self.m_grid2.SetCellValue(i, 3, self.logcat_datetime_in[i])
            self.m_grid2.SetCellValue(i, 4, self.logcat_late[i])
            self.m_grid2.SetCellValue(i, 5, self.logcat_datetime_out[i])
            self.m_grid2.SetCellValue(i, 6, self.logcat_early[i])

    def OnSelect2(self, e):
        s = e.GetString()
        self.dic['date'] = s
        print(self.dic)
        self.selectRow()
        self.m_grid2.ClearGrid()
        for i, id in enumerate(self.logcat_id):
            self.m_grid2.SetCellValue(i, 0, str(id))
            self.m_grid2.SetCellValue(i, 1, self.logcat_unit[i])
            self.m_grid2.SetCellValue(i, 2, self.logcat_name[i])
            self.m_grid2.SetCellValue(i, 3, self.logcat_datetime_in[i])
            self.m_grid2.SetCellValue(i, 4, self.logcat_late[i])
            self.m_grid2.SetCellValue(i, 5, self.logcat_datetime_out[i])
            self.m_grid2.SetCellValue(i, 6, self.logcat_early[i])

    def OnSelect3(self, e):
        s = e.GetString()
        self.dic['id'] = s
        print(self.dic)
        self.selectRow()
        self.m_grid2.ClearGrid()
        for i, id in enumerate(self.logcat_id):
            self.m_grid2.SetCellValue(i, 0, str(id))
            self.m_grid2.SetCellValue(i, 1, self.logcat_unit[i])
            self.m_grid2.SetCellValue(i, 2, self.logcat_name[i])
            self.m_grid2.SetCellValue(i, 3, self.logcat_datetime_in[i])
            self.m_grid2.SetCellValue(i, 4, self.logcat_late[i])
            self.m_grid2.SetCellValue(i, 5, self.logcat_datetime_out[i])
            self.m_grid2.SetCellValue(i, 6, self.logcat_early[i])

    def OnStatistic1(self, e):
        fr = wx.Frame(None, title='统计图', size=wx.Size(480, 280))
        panel = CanvasPanel(fr, self.origin)
        panel.DrawUnit()
        fr.Show()

    def OnStatistic2(self, e):
        fr = wx.Frame(None, title='统计图', size=wx.Size(480, 280))
        panel = CanvasPanel(fr, self.origin)
        panel.DrawDate()
        fr.Show()

    def OnStatistic3(self, e):
        fr = wx.Frame(None, title='统计图', size=wx.Size(480, 280))
        panel = CanvasPanel(fr, self.origin)
        panel.DrawId()
        fr.Show()

    def OnSave(self, e):

        with wx.FileDialog(self, "Save csv file", wildcard="csv files (*.csv)|*.csv",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            pathname = fileDialog.GetPath()
            df = pd.DataFrame(self.origin, columns=['学号', '学院', '姓名', '签到时间', '迟到', '签退时间', '早退'])
            df.to_csv(pathname, index=False)
            print('保存成功：%s' % pathname)


    def selectRow(self):
        conn = pymysql.connect('localhost', 'root', '', 'facedb')  # 建立数据库连接
        cur = conn.cursor()  # 得到游标对象

        sql_select = ''

        if self.dic['date'].strip() != '':
            if sql_select != '':
                sql_select += 'and '
            sql_select += '''datetime_in between '{}' and '{}' '''.format(self.dic['date'],
                                                                       (datetime.datetime.strptime(self.dic['date'], '%Y-%m-%d')+
                                                                        datetime.timedelta(days=1)).strftime('%Y-%m-%d'))

        if self.dic['id'].strip() != '':
            if sql_select != '':
                sql_select += 'and '
            sql_select += '''id={} '''.format(self.dic['id'])

        if self.dic['unit'].strip() != '':
            if sql_select != '':
                sql_select += 'and '
            sql_select += '''unit='{}' '''.format(self.dic['unit'])
        if sql_select.strip() != '':
            sql_select = 'where ' + sql_select
        cur.execute(
            '''select id, unit, name, datetime_in, late, datetime_out, early from logcat ''' + sql_select)
        self.origin = np.array(cur.fetchall())

        self.logcat_id, self.logcat_unit, self.logcat_name, self.logcat_datetime_in, self.logcat_late, self.logcat_datetime_out, self.logcat_early = [], [], [], [], [], [], []
        if self.origin != []:
            self.logcat_id, self.logcat_unit, self.logcat_name, self.logcat_datetime_in, self.logcat_late, self.logcat_datetime_out, self.logcat_early = self.origin.T
            self.logcat_id = [int(i) for i in self.logcat_id]

        cur.close()
        conn.commit()
        conn.close()


class CanvasPanel(wx.Panel):
    def __init__(self, parent, origin):
        wx.Panel.__init__(self, parent)
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        self.origin = origin
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def DrawUnit(self):
        df = pd.DataFrame(self.origin, columns=['总数', '学院', '姓名', '签到时间', '迟到', '签退时间', '早退'])

        early = df[df['迟到'] == '是'].groupby(["学院"], as_index=False)['迟到'].count()
        late = df[df['早退'] == '是'].groupby(["学院"], as_index=False)['早退'].count()
        total = df.groupby(["学院"], as_index=False)['总数'].count()

        late = pd.merge(total, late, how='left', on='学院')
        df = pd.merge(late, early, how='left', on='学院')

        df = df.fillna(0)

        df['迟到率'] = df['迟到'] / df['总数']
        df['早退率'] = df['早退'] / df['总数']

        # a = df[['学院', '迟到率', '早退率']].plot.bar(x='学院', rot=0)

        b1 = self.axes.bar(np.arange(1, len(df['学院'])+1,1)-0.15, df['迟到率'], width=0.3, align='center', label='迟到率')
        b2 = self.axes.bar(np.arange(1, len(df['学院'])+1,1)+0.15, df['早退率'], width=0.3, align='center', label='早退率')
        for b in b1+b2:
            h = b.get_height()
            self.axes.text(b.get_x()+b.get_width()/2, h, '%d%%' % int(h*100), ha='center', va='bottom')
        self.axes.set_ylim(0, 1.1)
        self.axes.set_xticks(np.arange(1, len(df['学院'])+1, 1))
        self.axes.set_xticklabels(list(df['学院']))
        self.axes.set_title('按学院统计迟到/早退率')
        self.axes.legend()


    def DrawId(self):
        df = pd.DataFrame(self.origin, columns=['总数', '学院', '姓名', '签到时间', '迟到', '签退时间', '早退'])

        early = df[df['迟到'] == '是'].groupby(["姓名"], as_index=False)['迟到'].count()
        late = df[df['早退'] == '是'].groupby(["姓名"], as_index=False)['早退'].count()
        total = df.groupby(["姓名"], as_index=False)['总数'].count()

        late = pd.merge(total, late, how='left', on='姓名')
        df = pd.merge(late, early, how='left', on='姓名')

        df = df.fillna(0)

        df['迟到率'] = df['迟到'] / df['总数']
        df['早退率'] = df['早退'] / df['总数']

        # a = df[['学院', '迟到率', '早退率']].plot.bar(x='学院', rot=0)


        b1 = self.axes.bar(np.arange(1, len(df['姓名'])+1,1)-0.15, df['迟到率'], width=0.3, align='center', label='迟到率')
        b2 = self.axes.bar(np.arange(1, len(df['姓名'])+1,1)+0.15, df['早退率'], width=0.3, align='center', label='早退率')
        for b in b1+b2:
            h = b.get_height()
            self.axes.text(b.get_x()+b.get_width()/2, h, '%d%%' % int(h*100), ha='center', va='bottom')
        self.axes.set_ylim(0, 1.1)
        self.axes.set_xticks(np.arange(1, len(df['姓名'])+1, 1))
        self.axes.set_xticklabels(list(df['姓名']))
        self.axes.set_title('按个人统计迟到/早退率')
        self.axes.legend()

    def DrawDate(self):
        df = pd.DataFrame(self.origin, columns=['总数', '学院', '姓名', '签到时间', '迟到', '签退时间', '早退'])

        for t in ['签到时间', '签退时间']:
            df[t] = df[t].apply(lambda x: x.split(' ')[0])

        early = df[df['迟到'] == '是'].groupby(["签到时间"], as_index=False)['迟到'].count()
        late = df[df['早退'] == '是'].groupby(["签到时间"], as_index=False)['早退'].count()
        total = df.groupby(["签到时间"], as_index=False)['总数'].count()

        late = pd.merge(total, late, how='left', on='签到时间')
        df = pd.merge(late, early, how='left', on='签到时间')

        df = df.fillna(0)

        df['迟到率'] = df['迟到'] / df['总数']
        df['早退率'] = df['早退'] / df['总数']

        # a = df[['学院', '迟到率', '早退率']].plot.bar(x='学院', rot=0)


        b1 = self.axes.bar(np.arange(1, len(df['签到时间'])+1,1)-0.15, df['迟到率'], width=0.3, align='center', label='迟到率')
        b2 = self.axes.bar(np.arange(1, len(df['签到时间'])+1,1)+0.15, df['早退率'], width=0.3, align='center', label='早退率')
        for b in b1+b2:
            h = b.get_height()
            self.axes.text(b.get_x()+b.get_width()/2, h, '%d%%' % int(h*100), ha='center', va='bottom')
        self.axes.set_ylim(0, 1.1)
        self.axes.set_xticks(np.arange(1, len(df['签到时间'])+1, 1))
        self.axes.set_xticklabels(list(df['签到时间']))
        self.axes.set_title('按日期统计迟到/早退率')
        self.axes.legend()