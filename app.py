############## Use below for production with manual metrics input from above

# Partha Pratim Ray

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import pandas as pd
import yaml

# ---------------------
# Configuration
# ---------------------

# Paths
model_path = "best.pt"       # Ensure the best.pt is in the local directory or provide full path
data_yaml_path = "data.yaml"  # Ensure data.yaml is in the local directory or provide full path

# Check if required files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")
if not os.path.exists(data_yaml_path):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}.")

# Load the YOLO model
model = YOLO(model_path)

# Load class names from data.yaml
with open(data_yaml_path, 'r') as stream:
    data_dict = yaml.safe_load(stream)
class_names = data_dict['names']  # e.g., ['Afghan_hound', 'African_hunting_dog', ...] up to 120 classes

# ---------------------
# Metrics Data
# ---------------------

# Overall Metrics
overall_metrics = {
    "Class": "Overall",
    "Precision": 0.7710520455384078,
    "Recall": 0.7396299270284923,
    "mAP50": 0.8090336605148044,
    "mAP50-95": 0.7178123217082027,
    "mAP75": 0.777247420215978
}

# Per-Class Metrics as a multi-line string
per_class_metrics_str = """
0: Precision=0.950051944721634, Recall=0.9259259259259259, mAP50=0.9755763198757764, mAP50-95=0.8761168602246491
1: Precision=0.913220142622241, Recall=0.7941176470588235, mAP50=0.8960616051262742, mAP50-95=0.7549103010429146
2: Precision=0.7372984951890349, Recall=0.7895319766491371, mAP50=0.8814292455141511, mAP50-95=0.8206840991617665
3: Precision=0.4590719866278696, Recall=0.5, mAP50=0.461591956701709, mAP50-95=0.4211189219475949
4: Precision=0.6184005433721002, Recall=0.55, mAP50=0.606004146196315, mAP50-95=0.512222940400535
5: Precision=0.5713923109334266, Recall=0.5516453651312371, mAP50=0.6077089403423442, mAP50-95=0.5480459050646989
6: Precision=0.7171550800348669, Recall=0.8076923076923077, mAP50=0.8277810374576291, mAP50-95=0.716992210820697
7: Precision=0.9159120375114393, Recall=0.9318181818181818, mAP50=0.9646808976669168, mAP50-95=0.8816416588132834
8: Precision=0.9913887843872787, Recall=0.896551724137931, mAP50=0.9622860780984719, mAP50-95=0.87970008325344
9: Precision=0.7340852070508719, Recall=0.613871078813217, mAP50=0.7510088541708057, mAP50-95=0.6775157428360266
10: Precision=0.8420572532183483, Recall=0.9642857142857143, mAP50=0.9420676470588236, mAP50-95=0.8492293322429207
11: Precision=0.7099820948850775, Recall=0.8611111111111112, mAP50=0.8270923840557883, mAP50-95=0.7008903529428722
12: Precision=0.8634064755508549, Recall=0.7901915514513194, mAP50=0.9023652651696131, mAP50-95=0.8518173491764902
13: Precision=0.9299135123785923, Recall=0.8947368421052632, mAP50=0.9382840909090909, mAP50-95=0.8506474227923493
14: Precision=0.8306860735970896, Recall=0.7619047619047619, mAP50=0.8547228953537712, mAP50-95=0.7928691056143748
15: Precision=0.7551169289450425, Recall=0.6429092513076484, mAP50=0.7430199755151243, mAP50-95=0.6916089837167693
16: Precision=0.7799482742887337, Recall=0.886157931584948, mAP50=0.8752104461316781, mAP50-95=0.7687241130428973
17: Precision=0.8575207045294126, Recall=0.42857142857142855, mAP50=0.7140262820249389, mAP50-95=0.6183897172277056
18: Precision=0.8716010553696922, Recall=0.7037037037037037, mAP50=0.8307099280692011, mAP50-95=0.7576428475073667
19: Precision=0.7631703486659736, Recall=0.71875, mAP50=0.8517840363359761, mAP50-95=0.7270017216968
20: Precision=0.7006786499565775, Recall=0.5357142857142857, mAP50=0.681074868661097, mAP50-95=0.6043264313001361
21: Precision=0.879003331472376, Recall=0.7727272727272727, mAP50=0.8806168979931869, mAP50-95=0.7667080346058525
22: Precision=0.7399740934714494, Recall=0.9473684210526315, mAP50=0.8270595091183327, mAP50-95=0.7297418045330895
23: Precision=0.6797523465134718, Recall=0.6634822741917626, mAP50=0.7290151202431302, mAP50-95=0.6389816563450246
24: Precision=0.5106575767372992, Recall=0.2692307692307692, mAP50=0.4628142655915253, mAP50-95=0.4259814227175367
25: Precision=0.7178351457649109, Recall=0.6363877593290892, mAP50=0.7636380572293818, mAP50-95=0.703198583575968
26: Precision=0.7683054771818867, Recall=0.76, mAP50=0.8446857768667896, mAP50-95=0.785944137555996
27: Precision=0.9118273620472525, Recall=0.9411764705882353, mAP50=0.9854761904761906, mAP50-95=0.9027474156118146
28: Precision=0.9397295768801787, Recall=0.9, mAP50=0.9599586372531578, mAP50-95=0.8124781140856718
29: Precision=0.5533603918415103, Recall=0.4850790559103108, mAP50=0.6570443199470123, mAP50-95=0.5567443986552293
30: Precision=0.7277265819014959, Recall=0.6341463414634146, mAP50=0.7712519423559233, mAP50-95=0.6501650478097549
31: Precision=0.9026604718725301, Recall=0.6956186570121078, mAP50=0.8680566835689678, mAP50-95=0.6978489639647801
32: Precision=0.8135026607058359, Recall=0.7522680538999782, mAP50=0.8354574758576219, mAP50-95=0.7821063558458735
33: Precision=0.7790779039480513, Recall=0.9, mAP50=0.8962941688074686, mAP50-95=0.804624933068968
34: Precision=0.6679276293654098, Recall=0.8, mAP50=0.7718791094285562, mAP50-95=0.6901626796134892
35: Precision=0.7498106300805424, Recall=1.0, mAP50=0.9288888888888888, mAP50-95=0.8281384745931912
36: Precision=0.669169980455674, Recall=0.6363636363636364, mAP50=0.7484207820518326, mAP50-95=0.6197496206486401
37: Precision=0.6608559782633446, Recall=0.47073650675174755, mAP50=0.7066760114303333, mAP50-95=0.5642940639926054
38: Precision=0.8076999229108593, Recall=0.891152802239241, mAP50=0.8391059986380347, mAP50-95=0.7108548810513233
39: Precision=0.7968481463986551, Recall=0.85, mAP50=0.9042326007326007, mAP50-95=0.8318236263736264
40: Precision=0.8023103515088807, Recall=0.8776654610627173, mAP50=0.896700952322643, mAP50-95=0.8317242069728058
41: Precision=0.5921871873135267, Recall=0.6129032258064516, mAP50=0.6671298280474146, mAP50-95=0.6157299090903454
42: Precision=0.9252749675177708, Recall=1.0, mAP50=0.995, mAP50-95=0.8966156368778817
43: Precision=0.6805277669157428, Recall=0.5992657872763302, mAP50=0.6030014064050425, mAP50-95=0.5489571828955201
44: Precision=0.8494201308178745, Recall=0.8585289528827474, mAP50=0.9241694421237407, mAP50-95=0.8100875509127746
45: Precision=0.7024809793928225, Recall=0.7874260309720877, mAP50=0.8845747233909929, mAP50-95=0.7949268449653824
46: Precision=0.7025574866255323, Recall=0.76, mAP50=0.8325492141035347, mAP50-95=0.666734865195336
47: Precision=0.7591941149453945, Recall=0.5912316092043576, mAP50=0.7512858105633272, mAP50-95=0.6764176572398436
48: Precision=0.8456657996345192, Recall=0.8430628848647751, mAP50=0.9399850452197447, mAP50-95=0.8786812002629617
49: Precision=0.7204710725917737, Recall=0.7352941176470589, mAP50=0.741237399809586, mAP50-95=0.6761715420054211
50: Precision=0.8253276396888147, Recall=0.8928571428571429, mAP50=0.906570616883117, mAP50-95=0.8174792122718552
51: Precision=0.8960256069175631, Recall=0.7368421052631579, mAP50=0.8653566969859104, mAP50-95=0.7222711865850817
52: Precision=0.8965837951145418, Recall=0.9642857142857143, mAP50=0.967472340425532, mAP50-95=0.8589636557920141
53: Precision=0.9545083115556321, Recall=0.8571428571428571, mAP50=0.9135674652406416, mAP50-95=0.8215518853555006
54: Precision=0.5646648682359179, Recall=0.44, mAP50=0.5619231870105881, mAP50-95=0.4908901394556514
55: Precision=0.7872131727479629, Recall=0.9254203918719266, mAP50=0.8099451754385967, mAP50-95=0.7379299742891713
56: Precision=0.9144822017339415, Recall=0.9583333333333334, mAP50=0.9758695652173913, mAP50-95=0.9004608041843742
57: Precision=0.7574007279689324, Recall=0.8147857737779668, mAP50=0.874966859820315, mAP50-95=0.8177803359690431
58: Precision=0.7840435303646889, Recall=0.9310344827586207, mAP50=0.9344552352166162, mAP50-95=0.8622790610302784
59: Precision=0.7905591604614023, Recall=0.9090909090909091, mAP50=0.9071769202766654, mAP50-95=0.8498133648608812
60: Precision=0.6393577679658906, Recall=0.71875, mAP50=0.7464381385789797, mAP50-95=0.7028502401944823
61: Precision=0.9636942143168822, Recall=0.926829268292683, mAP50=0.9690184704275709, mAP50-95=0.8587403761587502
62: Precision=0.7106422187027044, Recall=0.7372477654594655, mAP50=0.811168030248403, mAP50-95=0.6946423381867156
63: Precision=0.6240267051567073, Recall=0.7419354838709677, mAP50=0.6856916079641516, mAP50-95=0.5598098005606787
64: Precision=0.5708113423650005, Recall=0.4, mAP50=0.5670697751464068, mAP50-95=0.4623254566524631
65: Precision=0.712308984376764, Recall=0.6818181818181818, mAP50=0.6822830260727701, mAP50-95=0.6207656181810598
66: Precision=0.9281391491434923, Recall=0.9259259259259259, mAP50=0.9771428571428571, mAP50-95=0.9193733014191849
67: Precision=0.7822224074931787, Recall=0.8164048926504403, mAP50=0.9250230341178616, mAP50-95=0.8312403244806166
68: Precision=0.7407527943592982, Recall=0.5681818181818182, mAP50=0.7576132015540581, mAP50-95=0.6539397100680419
69: Precision=0.658344258202623, Recall=0.5238095238095238, mAP50=0.6335207628321592, mAP50-95=0.5049619631877108
70: Precision=0.8231282250313716, Recall=0.96, mAP50=0.9436381749870592, mAP50-95=0.8243550313423704
71: Precision=0.7804906787812523, Recall=0.7359250181530276, mAP50=0.8436523291064535, mAP50-95=0.7293180250825338
72: Precision=0.7636938036158667, Recall=0.8571428571428571, mAP50=0.9038575262456839, mAP50-95=0.8255486611843585
73: Precision=0.7210597817728865, Recall=0.7, mAP50=0.747461695975865, mAP50-95=0.6554385227231857
74: Precision=0.8373838694891609, Recall=0.7925992470528516, mAP50=0.8964220894115631, mAP50-95=0.7998322000142604
75: Precision=0.6948102089833058, Recall=0.6282619254581066, mAP50=0.7272340541915377, mAP50-95=0.6055805877651762
76: Precision=0.8140855787046279, Recall=0.8, mAP50=0.8444618372423327, mAP50-95=0.6635431207567797
77: Precision=0.7694040466984147, Recall=0.6451612903225806, mAP50=0.8016044789081129, mAP50-95=0.7220811021287906
78: Precision=0.822004684194274, Recall=0.8553170634500037, mAP50=0.9001247244440818, mAP50-95=0.8446800734714162
79: Precision=0.8756831442439423, Recall=0.7288492706478783, mAP50=0.8527018889105804, mAP50-95=0.7948548317962955
80: Precision=0.9559447197223753, Recall=0.7236009393173949, mAP50=0.885492314226582, mAP50-95=0.8071829181758705
81: Precision=0.8578978417079882, Recall=0.8695652173913043, mAP50=0.9292433783108446, mAP50-95=0.869065290805738
82: Precision=0.940395346893057, Recall=0.5262511451637173, mAP50=0.8308674837523942, mAP50-95=0.713777575976463
83: Precision=0.5062456909760218, Recall=0.42857142857142855, mAP50=0.5831563152337651, mAP50-95=0.5284580407093069
84: Precision=0.7821500223465658, Recall=0.8461538461538461, mAP50=0.8896114018712242, mAP50-95=0.7632809864880561
85: Precision=0.8570093036531184, Recall=0.768415750498518, mAP50=0.8641878940891257, mAP50-95=0.7345890173170971
86: Precision=0.9336806482609095, Recall=0.8533788315552395, mAP50=0.9293629662486024, mAP50-95=0.8249410235935531
87: Precision=0.9345734419563879, Recall=0.84, mAP50=0.8993961136902314, mAP50-95=0.751260995580089
88: Precision=0.8253700344920221, Recall=0.6785714285714286, mAP50=0.7737868128717942, mAP50-95=0.6945037476840215
89: Precision=0.5591822015168044, Recall=0.40816081205181687, mAP50=0.5186574574038987, mAP50-95=0.3913971952114973
90: Precision=0.8162174621443528, Recall=0.8078689489851244, mAP50=0.9182480184871491, mAP50-95=0.8416554549009767
91: Precision=0.8074441943237106, Recall=0.7857142857142857, mAP50=0.883115072856214, mAP50-95=0.7708640311640476
92: Precision=0.8762454169520998, Recall=0.8125, mAP50=0.8436440648483553, mAP50-95=0.8085127428001734
93: Precision=0.7744758889653982, Recall=0.68, mAP50=0.8269884444312363, mAP50-95=0.6915551946760756
94: Precision=0.711088127413241, Recall=0.6857142857142857, mAP50=0.7831013411889166, mAP50-95=0.7180543483756786
95: Precision=0.7906170853084622, Recall=0.8262159362325655, mAP50=0.8480347086698847, mAP50-95=0.7374124386653089
96: Precision=0.831835967000895, Recall=0.9333333333333333, mAP50=0.9323333333333333, mAP50-95=0.9043257561980965
97: Precision=0.9837569186329179, Recall=0.9642857142857143, mAP50=0.9768309859154928, mAP50-95=0.8283180424900264
98: Precision=0.7417411361822507, Recall=0.6391371206075025, mAP50=0.7498195995357477, mAP50-95=0.70808938406264
99: Precision=0.9904679353245135, Recall=0.8846153846153846, mAP50=0.9028627166719194, mAP50-95=0.752072056680778
100: Precision=0.8317414830228937, Recall=0.75, mAP50=0.7459513869872813, mAP50-95=0.6926176615354661
101: Precision=0.41978680894089854, Recall=0.4782608695652174, mAP50=0.48141022360551583, mAP50-95=0.4144050706681329
102: Precision=0.6446827675977833, Recall=0.7619047619047619, mAP50=0.8253548103548103, mAP50-95=0.7289570241595414
103: Precision=0.771803414775816, Recall=0.7958540537409124, mAP50=0.8893964079800443, mAP50-95=0.8319058711090973
104: Precision=0.46852444226198303, Recall=0.44097931046456407, mAP50=0.39952897726462094, mAP50-95=0.3147688996209689
105: Precision=0.6631411027359093, Recall=0.5106839320057245, mAP50=0.6733223280489881, mAP50-95=0.5746738298302899
106: Precision=1.0, Recall=0.6651217326058613, mAP50=0.7984452522197043, mAP50-95=0.6863382402967833
107: Precision=0.8327070961152427, Recall=0.96, mAP50=0.9694927536231884, mAP50-95=0.8618759209987303
108: Precision=0.8126512912365189, Recall=0.7992019378892692, mAP50=0.8333851819016381, mAP50-95=0.7120968967901217
109: Precision=0.5905790605724864, Recall=0.5501193036758621, mAP50=0.6811595925297115, mAP50-95=0.5671807467011729
110: Precision=0.8607857169580835, Recall=0.8421052631578947, mAP50=0.8541075996374958, mAP50-95=0.7472136130583998
111: Precision=0.7602089400199521, Recall=0.5769230769230769, mAP50=0.7627324274723889, mAP50-95=0.6905122694487801
112: Precision=0.7889751824977789, Recall=0.7692307692307693, mAP50=0.7564113973353781, mAP50-95=0.6488640540762379
113: Precision=0.6494792910272318, Recall=0.43239109804927656, mAP50=0.6031385443937921, mAP50-95=0.5494966164345454
114: Precision=0.5877517671416174, Recall=0.5, mAP50=0.5915767929438827, mAP50-95=0.498830414109515
115: Precision=0.6011481453277376, Recall=0.7894736842105263, mAP50=0.7358054903166529, mAP50-95=0.6513141815274721
116: Precision=0.8737558428352519, Recall=0.64, mAP50=0.7586255320667661, mAP50-95=0.7208942608628005
117: Precision=0.6146246520159415, Recall=0.6818181818181818, mAP50=0.7404198709608547, mAP50-95=0.6515161494565639
118: Precision=0.6601156119013557, Recall=0.6475174296603731, mAP50=0.6695776738813121, mAP50-95=0.5734412428124116
119: Precision=0.8467176165515974, Recall=0.8076923076923077, mAP50=0.90133166969147, mAP50-95=0.8275215828900926
"""

# Function to parse per-class metrics from the multi-line string
def parse_per_class_metrics(metrics_str):
    per_class_metrics = []
    lines = metrics_str.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue  # Skip empty lines
        # Split by colon to separate class index and metrics
        try:
            class_idx_part, metrics_part = line.split(':', 1)
            class_idx = int(class_idx_part.strip())
            # Extract metrics using split and strip
            metrics = {}
            metrics["Class Index"] = class_idx
            metrics_list = metrics_part.strip().split(', ')
            for metric in metrics_list:
                key, value = metric.split('=')
                key = key.strip()
                value = float(value.strip())
                metrics[key] = value
            per_class_metrics.append(metrics)
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(e)
    return per_class_metrics

# Parse the per-class metrics
parsed_per_class_metrics = parse_per_class_metrics(per_class_metrics_str)

# Map class indices to class names
per_class_metrics_data = []
for metric in parsed_per_class_metrics:
    class_idx = metric.pop("Class Index")
    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class{class_idx}"
    metric["Class"] = class_name
    per_class_metrics_data.append(metric)

# Combine Overall Metrics and Per-Class Metrics
metrics_data = [overall_metrics] + per_class_metrics_data

# Create Metrics DataFrame
metrics_df = pd.DataFrame(metrics_data, columns=["Class", "Precision", "Recall", "mAP50", "mAP50-95"])

# ---------------------
# Inference Functions
# ---------------------

def run_inference(img: np.ndarray, model):
    """
    Runs inference on the input image using the YOLO model.
    Returns the detection results and the annotated image.
    """
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run prediction with specified confidence and IoU thresholds
    results = model.predict(img_rgb, conf=0.25, iou=0.6)
    
    detections = []
    res = results[0]
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()  # Bounding box coordinates
            conf = float(boxes.conf[i])     # Confidence score
            cls_idx = int(boxes.cls[i])     # Class index
            class_name = class_names[cls_idx]  # Class name
            detections.append([class_name, conf, *xyxy])
    return detections, results

def draw_boxes(image: np.ndarray, detections):
    """
    Draws bounding boxes and labels on the image.
    """
    # Define a color palette for classes (BGR)
    palette = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        # Add more colors as needed
    ]
    num_colors = len(palette)

    for det in detections:
        class_name, conf, x1, y1, x2, y2 = det
        # Find class index
        try:
            cls_idx = class_names.index(class_name)
        except ValueError:
            cls_idx = 0  # Default to first color if class not found
        color = palette[cls_idx % num_colors]

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Prepare label
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw filled rectangle behind text
        cv2.rectangle(image, (int(x1), int(y1)-th-10), (int(x1)+tw, int(y1)), color, -1)

        # Put text above the bounding box
        cv2.putText(image, label, (int(x1), int(y1)-5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image

def process_image(image):
    """
    Processes the input image, runs inference, and prepares the outputs.
    """
    # Convert PIL Image to NumPy array
    img = np.array(image)
    
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Run classification inference
    detections, results = run_inference(img_bgr, model)

    # Draw bounding boxes on the image
    annotated_img = draw_boxes(img_bgr.copy(), detections)

    # Convert annotated image back to RGB
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    annotated_img_pil = Image.fromarray(annotated_img_rgb)

    # Create Detection Results DataFrame
    det_df = pd.DataFrame(detections, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"])

    # Return annotated image, detection results, and metrics table
    return annotated_img_pil, det_df, metrics_df

# ---------------------
# Gradio Interface
# ---------------------

with gr.Blocks() as demo:
    gr.Markdown("# YOLO Dog Breed Detection Web App")
    gr.Markdown("Upload an image of a dog, and the model will detect and classify the breed, displaying bounding boxes, confidence scores, and precomputed validation metrics.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Run Inference")
        with gr.Column():
            annotated_image = gr.Image(type="pil", label="Annotated Image")
            det_results = gr.DataFrame(label="Detection Results")
            metrics_table = gr.DataFrame(value=metrics_df, label="Validation Metrics")
    
    submit_btn.click(fn=process_image, inputs=input_image, outputs=[annotated_image, det_results, metrics_table])

demo.launch()

