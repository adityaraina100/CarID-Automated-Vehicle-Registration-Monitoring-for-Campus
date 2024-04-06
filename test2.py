# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import requests
import csv

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


RAPIDAPI_KEY = "YOUR_API_KEY"
class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        detected_plates = []
        for det in preds:
            for *xyxy, _, cls in det:
                if int(cls) == YOUR_LICENSE_PLATE_CLASS_INDEX:
                    detected_plates.append(xyxy)

        # Make API calls to retrieve owner details
        for plate_coords in detected_plates:
            plate_img = crop_plate_from_image(plate_coords, orig_img)
            plate_number = recognize_license_plate(plate_img)
            owner_details = get_owner_details_from_api(plate_number, RAPIDAPI_KEY)

            # Save owner details to CSV file
            save_owner_details_to_csv(owner_details)


        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

def recognize_license_plate(plate_img):
    # URL of the license plate recognition service
    recognition_url = "https://api.platerecognizer.com/v1/plate-reader/"

    # Add your API key here
    API_KEY = "7f14710da7msh6b5def85deee5c9p124d23jsn1ca412804156"

    # Prepare headers with API key
    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    # Convert plate image to bytes
    _, img_encoded = cv2.imencode('.jpg', plate_img)
    plate_img_bytes = img_encoded.tobytes()

    # Prepare data to send
    files = {
        'upload': ('plate.jpg', plate_img_bytes, 'image/jpeg')
    }

    # Make a POST request to the service
    response = requests.post(recognition_url, headers=headers, files=files)

    # Check if request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_json = response.json()
        
        # Extract the license plate number from the response
        plates = response_json.get('results', [])
        if plates:
            # Assuming only one plate is detected
            plate_number = plates[0].get('plate')
            return plate_number
    
    # Return None if no plate number is detected or request fails
    return None

def get_owner_details_from_api(plate_number, api_key):
    url = "https://rto-vehicle-information-verification-india.p.rapidapi.com/api/v1/rc/vehicleinfo"

    payload = {
        "reg_no": plate_number,
        "consent": "Y",
        "consent_text": "I hear by declare my consent agreement for fetching my information via AITAN Labs API"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "rto-vehicle-information-verification-india.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
