import os, urllib.request
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "palm_multioutput_model.keras")  # đặt cùng thư mục với app.py
IMG_SIZE   = 224  # input size của model multi-output (MobileNetV2)

MEDIAPIPE_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
MEDIAPIPE_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
CROP_SIZE = 200  # chỉ dùng để cắt ảnh minh họa (visualize), không dùng để predict
CLASSES   = ["ngan", "trung_binh", "dai"]
LABELS_VI = {
    "sinh_dao": "Sinh Đạo", "tam_dao": "Tâm Đạo",
    "tri_dao":  "Trí Đạo",  "su_nghiep": "Sự Nghiệp",
}
LABELS_EN = {
    "sinh_dao": "Life Line", "tam_dao": "Heart Line",
    "tri_dao":  "Head Line", "su_nghiep": "Fate Line",
}

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# ─── MEDIAPIPE INIT ───────────────────────────────────────────────
@st.cache_resource
def load_detector():
    if not os.path.exists(MEDIAPIPE_MODEL):
        with st.spinner("Đang tải MediaPipe model (~3MB)..."):
            urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_MODEL)
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


# ─── CNN MODEL (multi-output) ─────────────────────────────────────
@st.cache_resource
def load_model_single():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Không tìm thấy model: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ─── FUZZY ENGINE ─────────────────────────────────────────────────
@st.cache_resource
def build_fuzzy():
    universe = np.arange(0, 1.01, 0.01)
    score_u  = np.arange(0, 11, 1)

    inputs = {k: ctrl.Antecedent(universe, k)
              for k in ["sinh_dao", "tam_dao", "tri_dao", "su_nghiep"]}
    suc_khoe  = ctrl.Consequent(score_u, "suc_khoe",  defuzzify_method="centroid")
    tinh_cam  = ctrl.Consequent(score_u, "tinh_cam",  defuzzify_method="centroid")
    cong_danh = ctrl.Consequent(score_u, "cong_danh", defuzzify_method="centroid")

    for v in inputs.values():
        v["ngan"]       = fuzz.trimf(v.universe, [0.0, 0.0, 0.45])
        v["trung_binh"] = fuzz.trimf(v.universe, [0.25, 0.5, 0.75])
        v["dai"]        = fuzz.trimf(v.universe, [0.55, 1.0, 1.0])

    suc_khoe["kem"]          = fuzz.trimf(score_u, [0, 0, 4])
    suc_khoe["binh_thuong"]  = fuzz.trimf(score_u, [3, 5, 7])
    suc_khoe["tot"]          = fuzz.trimf(score_u, [6, 10, 10])
    tinh_cam["ly_tri"]       = fuzz.trimf(score_u, [0, 0, 4])
    tinh_cam["can_bang"]     = fuzz.trimf(score_u, [3, 5, 7])
    tinh_cam["cam_tinh"]     = fuzz.trimf(score_u, [6, 10, 10])
    cong_danh["bien_dong"]   = fuzz.trimf(score_u, [0, 0, 4])
    cong_danh["on_dinh"]     = fuzz.trimf(score_u, [3, 5, 7])
    cong_danh["thanh_cong"]  = fuzz.trimf(score_u, [6, 10, 10])

    sd, td, trd, sn = (inputs[k] for k in ["sinh_dao","tam_dao","tri_dao","su_nghiep"])
    rules = [
        ctrl.Rule(sd["ngan"],                      suc_khoe["kem"]),
        ctrl.Rule(sd["dai"],                       suc_khoe["tot"]),
        ctrl.Rule(sd["trung_binh"],                suc_khoe["binh_thuong"]),
        ctrl.Rule(sd["dai"] & sn["dai"],           suc_khoe["tot"]),
        ctrl.Rule(td["ngan"],                      tinh_cam["ly_tri"]),
        ctrl.Rule(td["dai"],                       tinh_cam["cam_tinh"]),
        ctrl.Rule(trd["ngan"] & td["trung_binh"],  tinh_cam["ly_tri"]),
        ctrl.Rule(trd["dai"]  & td["ngan"],        tinh_cam["ly_tri"]),
        ctrl.Rule(trd["dai"]  & td["dai"],         tinh_cam["can_bang"]),
        ctrl.Rule(td["trung_binh"],                tinh_cam["can_bang"]),
        ctrl.Rule(sn["ngan"],                      cong_danh["bien_dong"]),
        ctrl.Rule(sn["dai"],                       cong_danh["on_dinh"]),
        ctrl.Rule(sn["dai"]  & trd["dai"],         cong_danh["thanh_cong"]),
        ctrl.Rule(sn["ngan"] & trd["ngan"],        cong_danh["bien_dong"]),
        ctrl.Rule(sn["trung_binh"],                cong_danh["on_dinh"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


# ─── XỬ LÝ ẢNH ───────────────────────────────────────────────────
def safe_crop(cx, cy, w, h, img):
    x1 = max(0, cx - w // 2);  y1 = max(0, cy - h // 2)
    x2 = min(img.shape[1], cx + w // 2);  y2 = min(img.shape[0], cy + h // 2)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None, None
    return cv2.resize(patch, (CROP_SIZE, CROP_SIZE)), (x1, y1, x2, y2)


def rotate_upright(img_rgb, detector):
    h, w = img_rgb.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_img)
    if not res.hand_landmarks:
        return None, None
    lm = res.hand_landmarks[0]
    pt0 = np.array([lm[0].x * w, lm[0].y * h])
    pt9 = np.array([lm[9].x * w, lm[9].y * h])
    angle = 90 - np.degrees(np.arctan2(-(pt9[1]-pt0[1]), pt9[0]-pt0[0]))
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img_rgb, M, (w, h))
    mp_rot = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotated)
    res2 = detector.detect(mp_rot)
    if not res2.hand_landmarks:
        return None, None
    lm2 = np.array([[int(p.x*w), int(p.y*h)] for p in res2.hand_landmarks[0]])
    return rotated, lm2


def process_frame(img_rgb, detector):
    rotated, lm = rotate_upright(img_rgb, detector)
    if rotated is None:
        return None, None, None, "❌ Không phát hiện bàn tay! Hãy chụp lại với ánh sáng tốt hơn, đặt tay rõ ràng trong khung."

    ps = int(np.linalg.norm(lm[9] - lm[0]))
    bs = int(ps * 0.6)
    roi_defs = {
        "sinh_dao":  (int(0.35*lm[0][0]+0.45*lm[5][0]+0.20*lm[9][0]),
                      int(0.40*lm[0][1]+0.40*lm[5][1]+0.20*lm[9][1]),
                      int(ps*0.55), int(ps*0.65)),
        "tam_dao":   (int((lm[5][0]+lm[17][0])/2),
                      int(0.20*lm[0][1]+0.80*lm[9][1]), bs, bs),
        "tri_dao":   (int(0.60*lm[5][0]+0.40*lm[17][0]),
                      int(0.42*lm[0][1]+0.58*lm[9][1]),
                      int(ps*0.75), int(ps*0.45)),
        "su_nghiep": (int(0.65*lm[9][0]+0.35*lm[13][0]),
                      int(0.52*lm[0][1]+0.48*lm[9][1]),
                      int(ps*0.26), int(ps*0.52)),
    }
    # Crop chỉ dùng để MINH HỌA trực quan (không dùng để predict — model mới nhận toàn ảnh)
    crops, boxes = {}, {}
    for name, (cx, cy, w, h) in roi_defs.items():
        crop, box = safe_crop(cx, cy, w, h, rotated)
        if crop is None:
            return None, None, None, f"❌ Crop vùng '{name}' thất bại — tay nằm ngoài khung ảnh!"
        crops[name] = crop
        boxes[name] = box

    illus = rotated.copy()
    colors = {"sinh_dao":(255,80,80),"tam_dao":(80,200,80),
              "tri_dao":(80,80,255),"su_nghiep":(220,80,220)}
    labels_short = {"sinh_dao":"Sinh","tam_dao":"Tâm",
                    "tri_dao":"Trí","su_nghiep":"Nghiệp"}
    for name, box in boxes.items():
        x1,y1,x2,y2 = box
        cv2.rectangle(illus,(x1,y1),(x2,y2),colors[name],3)
        cv2.putText(illus, labels_short[name], (x1+4,y1+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[name], 2)

    # Ảnh input thực sự cho model multi-output: toàn bộ bàn tay đã xoay thẳng, resize 224x224
    model_input = cv2.resize(rotated, (IMG_SIZE, IMG_SIZE))

    return crops, illus, model_input, None


def predict_and_fuzzy(model_input, model, fuzzy_sim):
    x = np.expand_dims(model_input.astype(np.float32) / 255.0, axis=0)
    raw_preds = model.predict(x, verbose=0)

    # Một số phiên bản Keras trả về dict {output_name: array}, số khác trả về list theo thứ tự output_names
    if isinstance(raw_preds, dict):
        pred_map = {name: arr[0] for name, arr in raw_preds.items()}
    else:
        pred_map = {name: arr[0] for name, arr in zip(model.output_names, raw_preds)}

    probs = {}
    for name in LABELS_VI:  # đảm bảo đủ và đúng 4 key: sinh_dao, tam_dao, tri_dao, su_nghiep
        pred = pred_map[name]
        probs[name] = {CLASSES[i]: float(pred[i]) for i in range(3)}

    crisp = {k: (v["ngan"]*0.0 + v["trung_binh"]*0.5 + v["dai"]*1.0)
             for k, v in probs.items()}

    print("="*50)
    print("CNN PROBS:", probs)
    print("CRISP:", crisp)
    print("="*50)

    for k, v in crisp.items():
        fuzzy_sim.input[k] = float(np.clip(v, 0.01, 0.99))
    fuzzy_sim.compute()

    scores = {
        "suc_khoe":  fuzzy_sim.output["suc_khoe"],
        "tinh_cam":  fuzzy_sim.output["tinh_cam"],
        "cong_danh": fuzzy_sim.output["cong_danh"],
    }
    return probs, crisp, scores


def interpret_4_vung(probs):
    """Diễn giải trực tiếp từng vùng chỉ tay dựa trên xác suất CNN."""
    interpretations = {
        "sinh_dao": {
            "ngan": "Đường Sinh Đạo ngắn — năng lượng có giới hạn, cần chú ý nghỉ ngơi điều độ.",
            "trung_binh": "Đường Sinh Đạo trung bình — sức khỏe ổn định, cân bằng.",
            "dai": "Đường Sinh Đạo dài — sinh lực dồi dào, sức bền tốt.",
        },
        "tam_dao": {
            "ngan": "Đường Tâm Đạo ngắn — thiên về lý trí, ít bộc lộ cảm xúc.",
            "trung_binh": "Đường Tâm Đạo trung bình — cân bằng cảm xúc và lý trí.",
            "dai": "Đường Tâm Đạo dài — giàu cảm xúc, sống tình cảm và chân thành.",
        },
        "tri_dao": {
            "ngan": "Đường Trí Đạo ngắn — tư duy thực tế, quyết định nhanh.",
            "trung_binh": "Đường Trí Đạo trung bình — tư duy cân bằng giữa logic và sáng tạo.",
            "dai": "Đường Trí Đạo dài — tư duy sâu sắc, phân tích kỹ lưỡng.",
        },
        "su_nghiep": {
            "ngan": "Đường Sự Nghiệp ngắn — con đường công danh có nhiều biến động.",
            "trung_binh": "Đường Sự Nghiệp trung bình — sự nghiệp ổn định.",
            "dai": "Đường Sự Nghiệp dài — tiềm năng thành công lớn, bền vững.",
        },
    }
    results = []
    for key in ["sinh_dao", "tam_dao", "tri_dao", "su_nghiep"]:
        p = probs[key]
        dominant = max(p, key=p.get)
        confidence = p[dominant]
        text = interpretations[key][dominant]
        results.append({
            "vung": LABELS_VI[key],
            "vung_en": LABELS_EN[key],
            "ket_qua": dominant,
            "do_tin_cay": confidence,
            "dien_giai": text,
        })
    return results


# ─── VISUALIZE ────────────────────────────────────────────────────
def make_figure(crops, illus, probs, scores):
    # Bảng màu huyền bí: tím đêm sâu + vàng kim
    BG       = "#0B0612"
    PANEL_BG = "#1B1130"
    GOLD     = "#E8C26A"
    LAVENDER = "#C9A8FF"

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, 0:2])
    ax_main.imshow(illus)
    ax_main.set_title("✦ Bàn Tay & Vùng Huyền Bí ✦", color=GOLD, fontsize=12, fontweight="bold", fontstyle="italic")
    ax_main.axis("off")

    positions = [(0,2),(0,3),(1,0),(1,1)]
    crop_colors = ["#E85D75","#52D17C","#6FA8FF","#D17CE8"]
    for (r,c), key, color in zip(positions, list(LABELS_VI.keys()), crop_colors):
        ax = fig.add_subplot(gs[r,c])
        ax.imshow(crops[key])
        p = probs[key]
        ax.set_title(
            f"{LABELS_EN[key]} ({LABELS_VI[key]})\nN:{p['ngan']:.2f} TB:{p['trung_binh']:.2f} D:{p['dai']:.2f}",
            color=color, fontsize=9, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(PANEL_BG)

    ax_r = fig.add_subplot(gs[1,2:4], polar=True)
    ax_r.set_facecolor(PANEL_BG)
    vals = [scores["suc_khoe"], scores["tinh_cam"], scores["cong_danh"]]
    N = 3
    angles = [n/N*2*np.pi for n in range(N)] + [0]
    vals_p  = vals + [vals[0]]
    ax_r.plot(angles, vals_p, "o-", lw=2.5, color=GOLD, markerfacecolor=LAVENDER, markersize=8)
    ax_r.fill(angles, vals_p, alpha=0.35, color=LAVENDER)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(["Sức khỏe","Tình cảm","Sự nghiệp"],
                          color="white", fontsize=10, fontweight="bold")
    ax_r.set_ylim(0,10); ax_r.tick_params(colors="#999")
    ax_r.spines["polar"].set_color("#5A4A78")
    ax_r.set_title("✦ Vận Mệnh Tổng Hợp ✦", color=GOLD, fontsize=12, fontweight="bold", fontstyle="italic", pad=18)

    fig.patch.set_facecolor(BG)
    return fig


def interpret(scores):
    sk, tc, cd = scores["suc_khoe"], scores["tinh_cam"], scores["cong_danh"]
    lines = []
    if sk >= 7:
        lines.append(("✦ Sinh Lực", f"{sk:.1f}/10", "Năng lượng dồi dào, sinh lực bền bỉ.", "green"))
    elif sk >= 4.5:
        lines.append(("✦ Sinh Lực", f"{sk:.1f}/10", "Thể trạng trung bình, nên chú ý nghỉ ngơi.", "orange"))
    else:
        lines.append(("✦ Sinh Lực", f"{sk:.1f}/10", "Dễ mệt mỏi, cần chú ý sức khỏe nhiều hơn.", "red"))

    if tc >= 7:
        lines.append(("✦ Tình Cảm", f"{tc:.1f}/10", "Tình cảm mạnh mẽ, chân thành, trái tim dẫn lối.", "green"))
    elif tc >= 4.5:
        lines.append(("✦ Tình Cảm", f"{tc:.1f}/10", "Cân bằng lý trí và tình cảm.", "orange"))
    else:
        lines.append(("✦ Tình Cảm", f"{tc:.1f}/10", "Rất lý trí, quyết đoán, tập trung vào bản thân.", "blue"))

    if cd >= 7.5:
        lines.append(("✦ Công Danh", f"{cd:.1f}/10", "Con đường rộng mở, tiềm năng thành công lớn.", "green"))
    elif cd >= 4.5:
        lines.append(("✦ Công Danh", f"{cd:.1f}/10", "Ổn định, ít biến động, cuộc sống bình yên.", "orange"))
    else:
        lines.append(("✦ Công Danh", f"{cd:.1f}/10", "Nhiều thăng trầm, cần học cách thích nghi.", "red"))
    return lines


# ─── SƠ ĐỒ MINH HỌA 4 ĐƯỜNG CHỈ TAY ────────────────────────────────
PALM_LINE_COLORS = {
    "sinh_dao":  "#E85D75",
    "tam_dao":   "#52D17C",
    "tri_dao":   "#6FA8FF",
    "su_nghiep": "#D17CE8",
}

PALM_LINE_INFO = {
    "sinh_dao": {
        "vi": "Sinh Đạo", "en": "Life Line",
        "vi_tri": "Vòng cong ôm sát gốc ngón cái, chạy từ kẽ ngón cái–ngón trỏ xuống gần cổ tay.",
        "y_nghia": "Phản ánh sinh lực, sức khỏe thể chất và sức bền của một người.",
        "muc": {
            "Ngắn": "Năng lượng có giới hạn, cần chú ý nghỉ ngơi điều độ.",
            "Trung bình": "Sức khỏe ổn định, cân bằng.",
            "Dài": "Sinh lực dồi dào, sức bền tốt.",
        },
    },
    "tam_dao": {
        "vi": "Tâm Đạo", "en": "Heart Line",
        "vi_tri": "Đường nằm cao nhất, chạy ngang phía dưới các ngón tay, từ mép trụ (cạnh ngón út) sang phía ngón trỏ/giữa.",
        "y_nghia": "Phản ánh đời sống tình cảm, cách một người yêu thương và biểu lộ cảm xúc.",
        "muc": {
            "Ngắn": "Thiên về lý trí, ít bộc lộ cảm xúc.",
            "Trung bình": "Cân bằng cảm xúc và lý trí.",
            "Dài": "Giàu cảm xúc, sống tình cảm và chân thành.",
        },
    },
    "tri_dao": {
        "vi": "Trí Đạo", "en": "Head Line",
        "vi_tri": "Đường nằm giữa lòng bàn tay, bắt đầu gần gốc ngón cái và chạy ngang qua trung tâm bàn tay.",
        "y_nghia": "Phản ánh tư duy, cách suy nghĩ, ra quyết định và khả năng phân tích.",
        "muc": {
            "Ngắn": "Tư duy thực tế, quyết định nhanh.",
            "Trung bình": "Tư duy cân bằng giữa logic và sáng tạo.",
            "Dài": "Tư duy sâu sắc, phân tích kỹ lưỡng.",
        },
    },
    "su_nghiep": {
        "vi": "Sự Nghiệp", "en": "Fate Line",
        "vi_tri": "Đường dọc chạy từ gần giữa cổ tay hướng thẳng lên ngón giữa, cắt ngang lòng bàn tay.",
        "y_nghia": "Phản ánh con đường công danh, sự nghiệp và những biến động trong cuộc sống.",
        "muc": {
            "Ngắn": "Con đường công danh có nhiều biến động.",
            "Trung bình": "Sự nghiệp ổn định.",
            "Dài": "Tiềm năng thành công lớn, bền vững.",
        },
    },
}


def palm_diagram_svg():
    """Sơ đồ SVG minh họa bàn tay với 4 đường chỉ tay được đánh dấu màu."""
    c_sinh = PALM_LINE_COLORS["sinh_dao"]
    c_tam  = PALM_LINE_COLORS["tam_dao"]
    c_tri  = PALM_LINE_COLORS["tri_dao"]
    c_su   = PALM_LINE_COLORS["su_nghiep"]
    palm_outline = (
        'M120,440 C95,440 80,420 78,395 L75,260 C74,250 80,242 90,242 '
        'C98,242 104,249 105,258 L108,310 L108,120 C108,108 117,99 128,99 '
        'C139,99 148,108 148,120 L148,255 L150,70 C150,57 160,47 172,47 '
        'C184,47 194,57 194,70 L194,255 L198,55 C198,42 208,32 220,32 '
        'C232,32 242,42 242,55 L242,255 L255,90 C257,79 267,71 278,73 '
        'C289,75 296,86 294,97 L268,250 C275,300 280,360 270,400 '
        'C260,432 235,440 200,440 Z'
    )
    svg_lines = [
        '<svg viewBox="0 0 360 460" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;display:block;margin:0 auto">',
        '<defs><radialGradient id="palmGlow" cx="50%" cy="45%" r="65%">'
        '<stop offset="0%" stop-color="#241640"/><stop offset="100%" stop-color="#140A26"/>'
        '</radialGradient></defs>',
        f'<path d="{palm_outline}" fill="url(#palmGlow)" stroke="#E8C26A" stroke-width="2.5" stroke-opacity="0.55"/>',
        f'<path d="M148,155 C122,172 98,212 90,260 C84,303 92,352 112,398" fill="none" stroke="{c_sinh}" stroke-width="5" stroke-linecap="round" opacity="0.95"/>',
        f'<path d="M97,200 C145,184 200,180 250,172" fill="none" stroke="{c_tam}" stroke-width="5" stroke-linecap="round" opacity="0.95"/>',
        f'<path d="M102,232 C155,243 205,247 248,236" fill="none" stroke="{c_tri}" stroke-width="5" stroke-linecap="round" opacity="0.95"/>',
        f'<path d="M180,408 C177,340 174,275 172,215 C170,175 169,150 169,132" fill="none" stroke="{c_su}" stroke-width="5" stroke-linecap="round" opacity="0.95"/>',
        f'<circle cx="250" cy="172" r="5" fill="{c_tam}"/>',
        f'<circle cx="248" cy="236" r="5" fill="{c_tri}"/>',
        f'<circle cx="169" cy="132" r="5" fill="{c_su}"/>',
        f'<circle cx="112" cy="398" r="5" fill="{c_sinh}"/>',
        '</svg>',
    ]
    return "".join(svg_lines)


# ─── TAB: TÌM HIỂU CHỈ TAY ─────────────────────────────────────────
def render_guide_tab():
    st.markdown(
        "<p class='subtitle-text' style='margin-top:0'>Khám phá ý nghĩa và vị trí của 4 đường chỉ tay chính trên lòng bàn tay</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col_diagram, col_legend = st.columns([1, 1.1])

    with col_diagram:
        st.markdown(
            f'<div class="palm-card" style="padding:24px 16px">{palm_diagram_svg()}</div>',
            unsafe_allow_html=True,
        )

    with col_legend:
        st.markdown("##### ✦ Chú giải màu sắc")
        for key in ["sinh_dao", "tam_dao", "tri_dao", "su_nghiep"]:
            color = PALM_LINE_COLORS[key]
            info = PALM_LINE_INFO[key]
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:8px 0">
<span style="width:18px;height:18px;border-radius:50%;background:{color};
box-shadow:0 0 10px {color}99;flex-shrink:0"></span>
<span style="font-family:'Cinzel',serif;color:#F4E4B8;font-weight:700">{info['en']}</span>
<span class="vi-text" style="color:#8a7aa8;font-style:italic">({info['vi']})</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📜 Giải Thích Chi Tiết 4 Đường Chỉ Tay")

    for key in ["sinh_dao", "tam_dao", "tri_dao", "su_nghiep"]:
        color = PALM_LINE_COLORS[key]
        info = PALM_LINE_INFO[key]
        muc_html = "".join(
            f"<li style='margin-bottom:4px'><b style='color:#F4E4B8'>{muc}:</b> "
            f"<span class='vi-text'>{mota}</span></li>"
            for muc, mota in info["muc"].items()
        )
        st.markdown(f"""
<div class="palm-card" style="border-left:4px solid {color}">
<div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap">
<span style="font-family:'Cinzel',serif;font-size:1.15rem;font-weight:700;color:{color}">{info['en']}</span>
<span class="vi-text" style="color:#8a7aa8;font-style:italic;font-size:0.95rem">({info['vi']})</span>
</div>
<div class="vi-text" style="margin-top:10px;color:#cbbfe0">
<b style="color:#E8C26A">📍 Vị trí:</b> {info['vi_tri']}
</div>
<div class="vi-text" style="margin-top:6px;color:#cbbfe0">
<b style="color:#E8C26A">✦ Ý nghĩa:</b> {info['y_nghia']}
</div>
<ul class="vi-text" style="margin-top:8px;color:#b8a6d9;font-size:0.92rem;padding-left:20px">
{muc_html}
</ul>
</div>
""", unsafe_allow_html=True)

    st.caption("✦ Lưu ý: nội dung mang tính tham khảo, giải trí và chiêm tinh dân gian, không phải kết luận y khoa hay khoa học.")


# ─── TAB: PHÂN TÍCH CHỈ TAY ─────────────────────────────────────────
def render_analysis_tab():
    with st.spinner("Đang triệu hồi các vì tinh tú AI..."):
        detector  = load_detector()
        model     = load_model_single()
        fuzzy_sim = build_fuzzy()
    st.success("✦ Quả cầu tiên tri đã sẵn sàng!", icon="🔮")

    col_cam, col_result = st.columns([1.2, 1])

    # ── CỘT TRÁI: INPUT ────────────────────────────────
    with col_cam:
        st.subheader("🔮 Take a photo of your hand")
        st.info("💡 Đặt lòng bàn tay thẳng, đủ ánh sáng, cách camera 30-50cm, nền đơn giản")

        tab_cam, tab_upload = st.tabs(["🌙 Chụp từ Camera", "✨ Upload ảnh"])

        with tab_cam:
            snap = st.camera_input("Nhấn nút chụp", key="camera_widget")
            if snap:
                img_pil = Image.open(snap).convert("RGB")
                img_rgb = np.array(img_pil)

                test_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                test_res = detector.detect(test_mp)

                if test_res.hand_landmarks:
                    st.success("✅ Phát hiện bàn tay!")
                else:
                    st.warning("⚠️ Chưa rõ bàn tay trong ảnh. Có thể phân tích sẽ thất bại.")

                if st.button("🔮 GIẢI MÃ VẬN MỆNH", type="primary", use_container_width=True, key="btn_camera"):
                    # Xóa kết quả cũ trước khi gán ảnh mới
                    for k in ["captured_frame", "source"]:
                        st.session_state.pop(k, None)
                    st.session_state["captured_frame"] = img_rgb
                    st.session_state["source"] = "camera"
                    st.rerun()

        with tab_upload:
            uploaded = st.file_uploader("Chọn ảnh bàn tay", type=["jpg","jpeg","png"], key="uploader_widget")
            if uploaded:
                img_pil = Image.open(uploaded).convert("RGB")
                img_rgb = np.array(img_pil)
                st.image(img_rgb, caption="Ảnh đã upload", use_column_width=True)
                if st.button("🔮 GIẢI MÃ VẬN MỆNH", type="primary", use_container_width=True, key="btn_upload"):
                    for k in ["captured_frame", "source"]:
                        st.session_state.pop(k, None)
                    st.session_state["captured_frame"] = img_rgb
                    st.session_state["source"] = "upload"
                    st.rerun()

        if st.button("🗑️ Xóa kết quả hiện tại", use_container_width=True):
            for k in ["captured_frame", "source"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── CỘT PHẢI: KẾT QUẢ ───────────────────────────────
    with col_result:
        st.subheader("✦ Prophecy")

        if "captured_frame" not in st.session_state:
            st.markdown("""
<div class='mystic-empty'>
<div class='glyph'>🔮</div>
<p>Quả cầu còn tĩnh lặng...<br>Chụp ảnh hoặc upload để khai mở vận mệnh.</p>
</div>
""", unsafe_allow_html=True)
            return  # dừng lại, không render phần dưới

        img_rgb = st.session_state["captured_frame"]

        with st.spinner("Đang đọc những đường chỉ tay ẩn giấu..."):
            crops, illus, model_input, err = process_frame(img_rgb, detector)

        if err:
            st.error(err)
            return

        probs, crisp, scores = predict_and_fuzzy(model_input, model, fuzzy_sim)

        # ── Score cards tổng hợp ──
        for icon_label, score_str, desc, color in interpret(scores):
            color_map = {"green":"#52D17C","orange":"#E8A85A",
                         "red":"#E85D6B","blue":"#6FA8FF"}
            c = color_map.get(color, "#C9A8FF")
            st.markdown(f"""
<div class="score-card">
<span style="font-size:1.15rem;font-weight:700;color:{c};font-family:'Cinzel',serif">{icon_label}  {score_str}</span><br>
<span class="vi-text" style="color:#cbbfe0;font-size:0.95rem">{desc}</span>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # ── Phân tích chi tiết 4 vùng ──
        st.subheader("📜 Giải Mã 4 Đường Chỉ Tay")
        label_map_short = {"ngan": "Ngắn", "trung_binh": "Trung bình", "dai": "Dài"}

        for item in interpret_4_vung(probs):
            conf_pct = item["do_tin_cay"] * 100
            conf_color = "#52D17C" if conf_pct >= 60 else "#E8A85A" if conf_pct >= 40 else "#E85D6B"

            st.markdown(f"""
<div class="palm-card" style="border-left:4px solid {conf_color}">
<div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:6px">
<span>
<span style="font-weight:700;color:#F4E4B8;font-size:1.05rem;font-family:'Cinzel',serif;letter-spacing:0.5px">{item['vung_en']}</span>
<span class="vi-text" style="color:#8a7aa8;font-size:0.85rem;font-style:italic;margin-left:8px">({item['vung']})</span>
</span>
<span style="color:{conf_color};font-weight:700">{label_map_short[item['ket_qua']]} ({conf_pct:.0f}%)</span>
</div>
<div class="vi-text" style="color:#b8a6d9;font-size:0.9rem;margin-top:8px;font-style:italic">{item['dien_giai']}</div>
</div>
""", unsafe_allow_html=True)

            if conf_pct < 45:
                st.caption(f"⚠️ Độ tin cậy thấp ({conf_pct:.0f}%) — kết quả vùng này có thể không chính xác.")

        st.divider()

        # ── Biểu đồ tổng hợp ──
        fig = make_figure(crops, illus, probs, scores)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Chi tiết xác suất CNN ──
        with st.expander("🔭 Chi tiết xác suất CNN"):
            for key in LABELS_VI:
                p = probs[key]
                st.markdown(f"**{LABELS_EN[key]}** <span class='vi-text' style='color:#8a7aa8;font-style:italic'>({LABELS_VI[key]})</span>", unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Ngắn",   f"{p['ngan']:.3f}")
                c2.metric("Trung bình", f"{p['trung_binh']:.3f}")
                c3.metric("Dài",    f"{p['dai']:.3f}")


# ─── STREAMLIT UI ─────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Destiny in Your Hands",
        page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🔮",
        layout="wide",
    )

    # ── CSS: Chủ đề huyền bí — tím đêm sâu, ánh vàng kim, sao lấp lánh ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,500&family=Noto+Serif:ital,wght@0,400;0,600;1,500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Cormorant Garamond', 'Noto Serif', serif;
    }

    /* Văn bản tiếng Việt — luôn dùng Noto Serif để không lỗi dấu */
    .vi-text {
        font-family: 'Noto Serif', 'Cormorant Garamond', serif;
    }

    .stApp {
        background: radial-gradient(ellipse at top, #1c1033 0%, #0a0614 55%, #050309 100%);
        background-attachment: fixed;
    }

    /* lớp sao lấp lánh phủ toàn trang */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image:
            radial-gradient(1.5px 1.5px at 10% 20%, #E8C26A 60%, transparent 100%),
            radial-gradient(1px 1px at 25% 70%, #C9A8FF 60%, transparent 100%),
            radial-gradient(1.5px 1.5px at 60% 15%, #ffffff 50%, transparent 100%),
            radial-gradient(1px 1px at 80% 45%, #E8C26A 50%, transparent 100%),
            radial-gradient(1.5px 1.5px at 92% 80%, #C9A8FF 50%, transparent 100%),
            radial-gradient(1px 1px at 40% 90%, #ffffff 50%, transparent 100%),
            radial-gradient(1.5px 1.5px at 50% 50%, #E8C26A 40%, transparent 100%);
        opacity: 0.55;
        pointer-events: none;
        z-index: 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #170D2B 0%, #0B0614 100%);
        border-right: 1px solid #3A2A55;
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.5px;
    }

    .app-header {
        text-align: center;
        padding: 6px 0 2px 0;
    }
    .app-header img {
        width: 92px;
        filter: drop-shadow(0 0 18px rgba(232, 194, 106, 0.55));
        margin-bottom: 6px;
    }
    .title-text {
        font-family: 'Cinzel', serif;
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #E8C26A, #F4E4B8, #C9A8FF, #E8C26A);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        text-shadow: 0 0 30px rgba(201, 168, 255, 0.25);
        letter-spacing: 1px;
    }
    .subtitle-text {
        text-align: center;
        color: #B8A6D9;
        font-size: 1.05rem;
        font-style: italic;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    hr, .stDivider {
        border-color: #3A2A55 !important;
    }

    /* Score card tổng hợp */
    .score-card {
        background: linear-gradient(135deg, rgba(36,22,64,0.85), rgba(20,12,38,0.9));
        border-radius: 14px; padding: 18px 20px;
        border: 1px solid #4A3870;
        border-left: 4px solid #E8C26A;
        margin: 10px 0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    }

    /* Card phân tích chi tiết từng vùng */
    .palm-card {
        background: linear-gradient(135deg, rgba(27,17,48,0.9), rgba(15,9,28,0.92));
        border-radius: 12px; padding: 16px 18px; margin: 10px 0;
        border: 1px solid #3A2A55;
        box-shadow: 0 3px 14px rgba(0,0,0,0.3);
    }

    /* Khối info / placeholder */
    .mystic-empty {
        text-align:center; padding:70px 20px; color:#7A6A99;
        background: radial-gradient(circle, rgba(201,168,255,0.05), transparent 70%);
        border-radius: 16px;
        border: 1px dashed #3A2A55;
    }
    .mystic-empty .glyph { font-size:3.6rem; filter: drop-shadow(0 0 12px rgba(232,194,106,0.4)); }

    /* Nút bấm */
    .stButton > button {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.8px;
        border-radius: 10px !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6B3FA0, #C9A8FF) !important;
        border: 1px solid #E8C26A !important;
        color: #1a0f2e !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(201,168,255,0.35);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 26px rgba(232,194,106,0.6);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Cinzel', serif;
        color: #B8A6D9;
    }
    .stTabs [aria-selected="true"] {
        color: #E8C26A !important;
    }

    /* Expander */
    details {
        background: rgba(27,17,48,0.6) !important;
        border-radius: 10px !important;
        border: 1px solid #3A2A55 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header với logo ──
    if os.path.exists(LOGO_PATH):
        import base64
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
<div class="app-header">
<img src="data:image/png;base64,{logo_b64}" />
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="title-text">✦ Destiny in Your Hands ✦</p>', unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Vận mệnh nằm trong lòng bàn tay — Chụp ảnh để khám phá điều ẩn giấu</p>",
                unsafe_allow_html=True)
    st.divider()

    tab_analyze, tab_guide = st.tabs(["🔮 Phân Tích Chỉ Tay", "📖 Tìm Hiểu Chỉ Tay"])
    with tab_analyze:
        render_analysis_tab()
    with tab_guide:
        render_guide_tab()


if __name__ == "__main__":
    main()
