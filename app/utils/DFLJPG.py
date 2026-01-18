import pickle
import struct
import traceback
import cv2
import numpy as np


def normalize_channels(img, target_channels):
    if img.ndim == 2:
        img = img[..., None]

    h, w, c = img.shape

    if c == 1 and target_channels == 3:
        img = np.repeat(img, 3, axis=2)

    if c > target_channels:
        img = img[..., :target_channels]

    return img


class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = {}
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            chunks = []
            data_counter = 0
            inst_length = inst.length

            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack(
                    "BB", data[data_counter:data_counter+2]
                )
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F
                    if 0 <= n <= 7:
                        chunk_name = f"RST{n}"
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2

                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"

                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = f"APP{n}"

                if chunk_size is None:
                    chunk_size, = struct.unpack(
                        ">H", data[data_counter:data_counter+2]
                    )
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and not (
                        data[c] == 0xFF and data[c+1] == 0xD9
                    ):
                        c += 1
                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append({
                    "name": chunk_name,
                    "m_h": chunk_m_h,
                    "data": chunk_data,
                    "ex_data": chunk_ex_data,
                })

            inst.chunks = chunks
            return inst

        except Exception as e:
            raise Exception(f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw(filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk["name"] in ["SOF0", "SOF2"]:
                    d = chunk["data"]
                    _, height, width = struct.unpack(">BHH", d[:5])
                    inst.shape = (height, width, 3)

                elif chunk["name"] == "APP15":
                    if isinstance(chunk["data"], bytes):
                        inst.dfl_dict = pickle.loads(chunk["data"])

            return inst

        except Exception:
            return None

    def get_img(self):
        if self.img is None:
            self.img = cv2.imread(self.filename)
        return self.img

    # ============================================================
    # XSeg mask
    # ============================================================
    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get("xseg_mask", None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = img[..., None]

        return img.astype(np.float32) / 255.0

    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict["xseg_mask"] = None
            return

        mask_a = normalize_channels(mask_a, 1)
        img_data = np.clip(mask_a * 255, 0, 255).astype(np.uint8)

        ret, buf = cv2.imencode(".png", img_data)
        if not ret:
            raise Exception("Failed to encode XSeg mask")

        self.dfl_dict["xseg_mask"] = buf

    # ============================================================
    # landmarks（68点 or 5点）
    # ============================================================
    def get_landmarks(self):
        lm = self.dfl_dict.get("landmarks", None)
        if lm is None:
            return None
        return np.array(lm, dtype=np.float32)

    def set_landmarks(self, lm):
        if lm is None:
            self.dfl_dict["landmarks"] = None
        else:
            self.dfl_dict["landmarks"] = np.asarray(lm, dtype=np.float32).tolist()

    # ============================================================
    # affine（image_to_face_mat）
    # ============================================================
    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get("image_to_face_mat", None)
        if mat is None:
            return None
        return np.array(mat, dtype=np.float32)

    def set_image_to_face_mat(self, mat):
        if mat is None:
            self.dfl_dict["image_to_face_mat"] = None
        else:
            self.dfl_dict["image_to_face_mat"] = np.asarray(mat, dtype=np.float32).tolist()

    # ============================================================
    # 保存（APP15 に dfl_dict を埋め込んで書き戻す）
    # ============================================================
    def save(self, filepath=None):
        """
        self.chunks を元に JPG を再構築しつつ、
        APP15 に self.dfl_dict を pickle して埋め込んで保存する。
        """
        if filepath is None:
            filepath = self.filename

        # dfl_dict を APP15 用バイト列に
        app15_data = pickle.dumps(self.dfl_dict) if self.dfl_dict is not None else b""
        app15_chunk = {
            "name": "APP15",
            "m_h": 0xEF,          # APP15 マーカー
            "data": app15_data,
            "ex_data": None,
        }

        # 既存の APP15 を削除しつつ、新しい APP15 を SOS の直前に挿入
        new_chunks = []
        app15_inserted = False

        for ch in self.chunks:
            name = ch["name"]

            # 既存の APP15 は捨てる
            if name == "APP15":
                continue

            # SOS の直前で APP15 を挿入
            if not app15_inserted and name == "SOS":
                new_chunks.append(app15_chunk)
                app15_inserted = True

            new_chunks.append(ch)

        # もし SOS が無かった場合は、とりあえず末尾に APP15 を追加
        if not app15_inserted:
            new_chunks.append(app15_chunk)

        # バイナリ再構築
        out = bytearray()
        for ch in new_chunks:
            m_h = ch["m_h"]
            data = ch["data"]
            ex_data = ch["ex_data"]

            # マーカー
            out += struct.pack("BB", 0xFF, m_h)

            # サイズ + データ
            if data is not None:
                size = len(data) + 2
                out += struct.pack(">H", size)
                out += data
            else:
                # サイズ無しのマーカー（SOI, EOI, RSTn など）
                pass

            # SOS の後ろの生データ
            if ex_data is not None:
                out += ex_data

        with open(filepath, "wb") as f:
            f.write(out)

        # 元データも更新しておく
        self.data = bytes(out)
        self.length = len(self.data)
        self.chunks = new_chunks
