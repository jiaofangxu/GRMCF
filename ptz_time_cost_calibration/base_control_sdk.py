from device_driver.device_init import Device
from device_driver.HCNetSDK import *
point_frame = NET_DVR_POINT_FRAME()
point_frame_ptr = LPNET_DVR_POINT_FRAME()


def start_up(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, TILT_UP, 0)  # 最后一个参数，0是启动，1是停止
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_up(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, TILT_UP, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_down(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, TILT_DOWN, 0)
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_down(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, TILT_DOWN, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_right(self):
    print(self.lUserID)
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, PAN_RIGHT, 0)
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_right(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, PAN_RIGHT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, PAN_LEFT, 0)
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, PAN_LEFT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_up_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, UP_LEFT, 0)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def stop_up_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, UP_LEFT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_down_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, DOWN_LEFT, 0)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def stop_down_left(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, DOWN_LEFT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_up_right(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, UP_RIGHT, 0)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def stop_up_right(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, UP_RIGHT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_down_right(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, DOWN_RIGHT, 0)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def stop_down_right(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, DOWN_RIGHT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_zoomin(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, ZOOM_IN, 0)
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_zoomin(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, ZOOM_IN, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')


def start_zoomout(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, ZOOM_OUT, 0)
    if lRet == 0:
        print('Start ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Start ptz control success')


def stop_zoomout(self):
    lRet = self.Objdll.NET_DVR_PTZControl_Other(self.lUserID, 1, ZOOM_OUT, 1)
    if lRet == 0:
        print('Stop ptz control fail, error code is: %d' % self.Objdll.NET_DVR_GetLastError())
    else:
        print('Stop ptz control success')












