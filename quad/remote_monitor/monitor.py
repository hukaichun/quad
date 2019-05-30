import struct
import socket
from mavlink.mavcrc import x25crc
import numpy as np




class ControlInfo:
    id = 1
    unpacker = struct.Struct('<Q4f3f3f3f3f4f3f3f1f')
    crc_extra = 35
    def __init__(self,buf):
        self._timestamp = buf[0]
        self._info = np.asarray(buf[1:-1])
        self._voltage = buf[-1]
        
    @property
    def attitude(self):
        return self._info[:13]

    @property
    def local_position(self):
        return self._info[20:23]

    @property
    def position_sp(self):
        return self._info[23:26]
    
    
    
    



#assume no signature
class MavlinkListener:
    header_unpacker          = struct.Struct('<cBBBBBBHB')
    checksum_unpacker        = struct.Struct('<H') 
    mavlink2_heaker_len      = 10

    def __init__(self,port):
        self.addr = ("0.0.0.0", port)
        self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_fd.bind(self.addr)


    def recv(self):
        while True:
            buf, addr = self.sock_fd.recvfrom(1500)
            if(len(buf)>0 and buf[0] == 0xfd):
                header = buf[:MavlinkListener.mavlink2_heaker_len]
                payload = buf[MavlinkListener.mavlink2_heaker_len:MavlinkListener.mavlink2_heaker_len+header[1]]
                check_sum = buf[MavlinkListener.mavlink2_heaker_len+header[1]:]
                
                crc_buf = bytearray(buf[1:-2])
                crc_buf.append(ControlInfo.crc_extra)
                crc = x25crc(crc_buf)

                magic, mlen, incompat_flags, compat_flags, seq, srcSystem, srcComponent, msgIdlow, msgIdhigh = MavlinkListener.header_unpacker.unpack(header)
                msgId = msgIdlow | (msgIdhigh<<16)
                if ControlInfo.id == msgId:

                    check_sum = MavlinkListener.checksum_unpacker.unpack(check_sum)
                    full_size = ControlInfo.unpacker.size
                    if header[1] < full_size:
                        payload = bytearray(payload)
                        payload.extend([0]*(full_size-header[1]))
                    payload = ControlInfo.unpacker.unpack(payload)

                    if check_sum[0] == crc.crc:
                        return ControlInfo(payload)
            



if __name__ == "__main__":
    from quad.gym import hover_2018
    s = MavlinkListener(8889)
    q = hover_2018.Quadrotors(init_num=1)
    q.render()
    count = 0
    tmp_time = 0
    while True:
        info = s.recv()
        count+=1

        print(info._timestamp - tmp_time)
        tmp_time = info._timestamp
        if count%20 == 0:
            q.attitude = info.attitude
            q.position = info._info[20:23]
            q.mark = info.local_position
            q.render()
            count = 0
            print("recved")

