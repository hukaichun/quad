import struct
import socket

class MavlinkListener:
	header_unpacker          = struct.Struct('<cBBBBBBHB')
	checksum_unpacker = struct.Struct('<H') 

	mavlink2_heaker_len        = 10

	def __init__(self,port):
		self.addr = ("0.0.0.0", port)
		self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock_fd.bind(self.addr)


	def recv(self):
		buf, addr = self.sock_fd.recvfrom(1500)
		if(len(buf)>0 and buf[0] == 0xfd):
			header = buf[:MavlinkListener.mavlink2_heaker_len]
			payload = buf[MavlinkListener.mavlink2_heaker_len:MavlinkListener.mavlink2_heaker_len+header[1]]
			check_sum = buf[MavlinkListener.mavlink2_heaker_len+header[1]:]

			header = MavlinkListener.header_unpacker.unpack(header)
			check_sum = MavlinkListener.checksum_unpacker.unpack(check_sum)

			unpacker = struct.Struct('<Q4f3f3f3f3f4f3f3f1f')
			full_size = unpacker.size

			if header[1] < full_size:
				payload = bytearray(payload)
				payload.extend([0]*(full_size-header[1]))

			payload = unpacker.unpack(payload)


		return (header, payload, check_sum)
			



if __name__ == "__main__":
	s = MavlinkListener(8889)

	for _ in range(255):
		h,p,c = s.recv()
		print("header :{}\n"
			  "payload: {}\n"
			  "check:{}".format(h,p,c))


