import pygame
import sys
import os
import random
from collections import deque
import time
import threading
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Ultralytics n'est pas installé. Les fonctionnalités de détection seront désactivées.")
    print("Pour installer: pip install ultralytics")
    YOLO_AVAILABLE = False

pygame.init()

# Constantes
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 15
CONVEYOR_WIDTH = 980
CONVEYOR_HEIGHT = 550
PAUSE_BUTTON_WIDTH = 120
PAUSE_BUTTON_HEIGHT = 40
OBSERVATION_WIDTH = 500
OBSERVATION_HEIGHT = 120
DETECTION_INTERVAL = 0.01
SCROLL_SPEED = 2 # Vitesse de défilement (pixels par frame)
IMAGE_HEIGHT_RATIO = 0.67
MIN_IMAGE_SPACING = 20 # Espacement minimum entre les images (en pixels)

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (235, 245, 255)
GREEN = (0, 180, 0)
RED = (220, 0, 0)
BUTTON_COLOR = (40, 45, 75)
BUTTON_HOVER_COLOR = (100, 120, 200)
BUTTON_TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (120, 140, 180)
PAUSE_BUTTON_COLOR = (80, 90, 120)
PAUSE_BUTTON_HOVER_COLOR = (120, 140, 200)
OBS_BACKGROUND = (250, 250, 255)

# Couleurs pour les boîtes de détection
DETECTION_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128)
]

# Création de la fenêtre
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Scanner de Bagages - Détection d'Objets Dangereux")

# Polices
font_title = pygame.font.SysFont('Arial', 22, bold=True)
font_button = pygame.font.SysFont('Arial', 16, bold=True)
font_status = pygame.font.SysFont('Arial', 18)
font_obs = pygame.font.SysFont('Arial', 16)
font_obs_title = pygame.font.SysFont('Arial', 16, bold=True)
font_detection = pygame.font.SysFont('Arial', 14, bold=True)


class ObjectDetector:

    def __init__(self, model_dir="model"):
        self.model = None
        self.model_dir = model_dir
        self.class_names = {}
        self.current_detections = []
        self.last_detection_time = 0
        self.is_detecting = False
        self.detection_lock = threading.Lock()
        self.detection_cache = {}
        self.detection_thread = None
        self.detection_queue = []

        self.load_model()

    def detect_for_path(self, image, image_path):
        #Détecte une fois pour cette image et stocke en cache
        if image_path in self.detection_cache:
            return self.detection_cache[image_path]

        print(f"Détection pour {image_path}, dimensions: {image.size if hasattr(image, 'size') else 'inconnues'}")
        detections = self.detect_objects(image)

        # Pour chaque détection, garder les coordonnées originales
        for det in detections:
            # Ces valeurs seront utilisées pour le redimensionnement plus tard
            det['x1_orig'] = det['x1']
            det['y1_orig'] = det['y1']
            det['x2_orig'] = det['x2']
            det['y2_orig'] = det['y2']

            # Coordonnées relatives (pour l'instant identiques aux originales)
            det['x1_rel'] = det['x1']
            det['y1_rel'] = det['y1']
            det['x2_rel'] = det['x2']
            det['y2_rel'] = det['y2']

        self.detection_cache[image_path] = detections
        print(f"Détections stockées pour {image_path}: {len(detections)} objets")
        return detections

    def load_model(self):
        #Charge le modèle YOLO et les noms de classes
        if not YOLO_AVAILABLE:
            print("YOLO n'est pas disponible. Impossible de charger le modèle.")
            return False

        try:
            model_path = os.path.join(self.model_dir, "model_exported.pt")
            model_onnx_path = os.path.join(self.model_dir, "model_exported.onnx")

            if os.path.exists(model_onnx_path):
                print(f"Chargement du modèle ONNX depuis {model_onnx_path}...")
                self.model = YOLO(model_onnx_path, task='detect')
            elif os.path.exists(model_path):
                print(f"Chargement du modèle PT depuis {model_path}...")
                self.model = YOLO(model_path, task='detect')
            else:
                print(f"Modèle non trouvé ni en .pt ni en .onnx")
                print("Tentative de chargement du modèle YOLOv8m par défaut...")
                self.model = YOLO("yolov8m.pt")  # Modèle par défaut

            # Charger les noms de classes depuis le fichier s'il existe
            classes_path = os.path.join(self.model_dir, "classes.txt")
            if os.path.exists(classes_path):
                print(f"Chargement des classes depuis {classes_path}...")
                self.class_names = {}
                with open(classes_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            idx = int(parts[0])
                            name = parts[1]
                            self.class_names[idx] = name
                print(f"Classes chargées: {self.class_names}")
            else:
                # Utiliser les noms par défaut du modèle
                self.class_names = self.model.names
                print(f"Utilisation des classes du modèle: {self.class_names}")

            print(f"Modèle chargé avec succès.")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            return False

    def detect_objects(self, image):
        #Détecte les objets dans une image
        if self.model is None:
            return []

        try:
            # Convertir l'image PIL en numpy pour YOLO
            img_array = np.array(image)
            print(f"\n--- NOUVELLE DÉTECTION ---\nDimensions de l'image: {img_array.shape}")

            results = self.model(img_array, conf=0.3)

            detections = []
            all_boxes = []
            for result in results:
                all_boxes.extend(result.boxes)

            if all_boxes:
                print("Résultats de la détection:")
                for result in results:
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.class_names.get(cls_id, f"Classe {cls_id}")
                        print(f"  Objet {i + 1}: {cls_name} (conf: {conf:.2f}) de [{x1},{y1}] à [{x2},{y2}]")
                        detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'color': DETECTION_COLORS[cls_id % len(DETECTION_COLORS)]
                        })
            else:
                print("Aucune détection.")

            return detections

        except Exception as e:
            print(f"Erreur lors de la détection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def update_detections(self, image, image_x, image_y):
        #Met à jour les détections avec une nouvelle image
        if image is None:
            return []

        # Vérifier l'intervalle de détection
        current_time = time.time()
        if current_time - self.last_detection_time < DETECTION_INTERVAL:
            return self.current_detections

        # Si aucun thread de détection n'est en cours, en lancer un nouveau
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.last_detection_time = current_time

            # Créer et démarrer un nouveau thread de détection
            self.detection_thread = threading.Thread(
                target=self._run_detection,
                args=(image, image_x, image_y)
            )
            self.detection_thread.daemon = True  # Le thread s'arrêtera quand le programme principal se termine
            self.detection_thread.start()

        return self.current_detections

    def _run_detection(self, image, image_x, image_y):
        # Fonction exécutée dans un thread séparé pour la détection
        try:
            detections = self.detect_objects(image)

            if detections:
                # Obtenir les dimensions de l'image
                if hasattr(image, 'size'):
                    img_width, img_height = image.size
                else:
                    img_width, img_height = 300, 200

                # Calculer le facteur d'échelle
                target_height = int(CONVEYOR_HEIGHT * 0.67)
                scale_factor = target_height / img_height

                # Ajuster les coordonnées
                for detection in detections:
                    # Appliquer l'échelle
                    detection['x1'] = int(detection['x1'] * scale_factor)
                    detection['y1'] = int(detection['y1'] * scale_factor)
                    detection['x2'] = int(detection['x2'] * scale_factor)
                    detection['y2'] = int(detection['y2'] * scale_factor)

                    # Appliquer le décalage
                    detection['x1_rel'] = detection['x1']
                    detection['y1_rel'] = detection['y1']
                    detection['x2_rel'] = detection['x2']
                    detection['y2_rel'] = detection['y2']

            # Mettre à jour les détections de manière thread-safe
            with self.detection_lock:
                self.current_detections = detections

        except Exception as e:
            print(f"Erreur dans le thread de détection: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_detections(self):
        with self.detection_lock:
            return self.current_detections.copy()

class Button:

    def __init__(self, x, y, width, height, text, action=None, color=BUTTON_COLOR, hover_color=BUTTON_HOVER_COLOR,
                     text_color=BUTTON_TEXT_COLOR):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False

    def draw(self, surface):
        current_color = self.hover_color if self.is_hovered else self.color

        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)

        border_color = (150, 180, 255) if self.is_hovered else (60, 70, 100)
        border_width = 3 if self.is_hovered else 2
        pygame.draw.rect(surface, border_color, self.rect, border_width, border_radius=8)

        text_surface = font_button.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def update(self, mouse_pos):
        x, y = mouse_pos
        self.is_hovered = (self.rect.left <= x <= self.rect.right and
                               self.rect.top <= y <= self.rect.bottom)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class ToggleButton(Button):

    def __init__(self, x, y, width, height, text_on, text_off, action=None, color=BUTTON_COLOR,
                 hover_color=BUTTON_HOVER_COLOR, text_color=BUTTON_TEXT_COLOR):
        super().__init__(x, y, width, height, text_off, action, color, hover_color, text_color)
        self.text_on = text_on
        self.text_off = text_off
        self.is_active = False

    def toggle(self):
        self.is_active = not self.is_active
        if self.action:
            self.action(self.is_active)

    def draw(self, surface):
        if self.is_active:
            current_color = (100, 180, 100) if self.is_hovered else (70, 150, 70)
            text = self.text_on
        else:
            current_color = self.hover_color if self.is_hovered else self.color
            text = self.text_off

        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)

        border_color = (150, 180, 255) if self.is_hovered else (60, 70, 100)
        border_width = 3 if self.is_hovered else 2
        pygame.draw.rect(surface, border_color, self.rect, border_width, border_radius=8)

        text_surface = font_button.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.toggle()
                return True
        return False


class ObservationPanel:

    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.border_rect = pygame.Rect(x - 2, y - 2, width + 4, height + 4)
        self.title = "Observations"
        self.detection_list = []
        self.max_detections = 3  # Limiter à 3 détections dans l'observation

    def update_detections(self, detections):
        if detections:
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            self.detection_list = []
            for det in sorted_detections[:self.max_detections]:
                self.detection_list.append({
                    'line1': "Objet potentiellement",
                    'line2': f"dangereux : {det['confidence']:.2f}",
                    'confidence': det['confidence']
                })
        else:
            self.detection_list = []

    def draw(self, surface):
        pygame.draw.rect(surface, BORDER_COLOR, self.border_rect, border_radius=8)
        pygame.draw.rect(surface, OBS_BACKGROUND, self.rect, border_radius=6)

        title_surface = font_obs_title.render(self.title, True, DARK_GRAY)
        title_rect = title_surface.get_rect(x=self.rect.x + 10, y=self.rect.y + 8)
        surface.blit(title_surface, title_rect)

        if self.detection_list:
            clear_text = "[Effacer]"
            clear_surface = font_obs.render(clear_text, True, (180, 0, 0))
            clear_rect = clear_surface.get_rect(
                x=self.rect.x + self.rect.width - clear_surface.get_width() - 10,
                y=self.rect.y + 8
            )
            surface.blit(clear_surface, clear_rect)

        pygame.draw.line(surface, DARK_GRAY,
                         (self.rect.x + 5, self.rect.y + 30),
                         (self.rect.x + self.rect.width - 5, self.rect.y + 30),
                         1)

        if self.detection_list:
            y_offset = 40
            for detection in self.detection_list:
                line1_surface = font_obs.render(detection['line1'], True, RED)
                line1_rect = line1_surface.get_rect(x=self.rect.x + 10, y=self.rect.y + y_offset)
                surface.blit(line1_surface, line1_rect)

                line2_surface = font_obs.render(detection['line2'], True, RED)
                line2_rect = line2_surface.get_rect(x=self.rect.x + 15, y=self.rect.y + y_offset + 20)
                surface.blit(line2_surface, line2_rect)

                y_offset += 45
        else:
            status_text = "Aucun objet détecté"
            status_surface = font_obs.render(status_text, True, GREEN)
            status_rect = status_surface.get_rect(x=self.rect.x + 10, y=self.rect.y + 50)
            surface.blit(status_surface, status_rect)

    def clear_detections(self):
        self.detection_list = []

    def click_handler(self, pos):
        if self.detection_list:
            clear_text = "[Effacer]"
            clear_surface = font_obs.render(clear_text, True, (180, 0, 0))
            clear_rect = pygame.Rect(
                self.rect.x + self.rect.width - clear_surface.get_width() - 10,
                self.rect.y + 8,
                clear_surface.get_width(),
                clear_surface.get_height()
            )

            if clear_rect.collidepoint(pos):
                self.clear_detections()
                return True
        return False

class ConveyorDisplay:

    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.border_rect = pygame.Rect(x - 4, y - 4, width + 8, height + 8)

        self.mask_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.mask_surface.fill((0, 0, 0, 0))

        self.background_surface = pygame.Surface((width, height))
        self.background_surface.fill(WHITE)

    def draw(self, surface):
        pygame.draw.rect(surface, BORDER_COLOR, self.border_rect, border_radius=10)

        pygame.draw.rect(surface, WHITE, self.rect, border_radius=8)

        pygame.draw.line(surface, GRAY,
                         (self.rect.x + 4, self.rect.y + self.rect.height - 4),
                         (self.rect.x + self.rect.width - 4, self.rect.y + self.rect.height - 4),
                         3)
        pygame.draw.line(surface, GRAY,
                         (self.rect.x + self.rect.width - 4, self.rect.y + 4),
                         (self.rect.x + self.rect.width - 4, self.rect.y + self.rect.height - 4),
                         3)


class StatusLight:

    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.is_red = False  # Vert par défaut

    def draw(self, surface):
        color = RED if self.is_red else GREEN

        for i in range(5, 0, -1):
            alpha = 30 - i * 5
            s = pygame.Surface((self.radius * 2 + i * 4, self.radius * 2 + i * 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color + (alpha,), (self.radius + i * 2, self.radius + i * 2), self.radius + i * 2)
            surface.blit(s, (self.x - self.radius - i * 2, self.y - self.radius - i * 2))

        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)

        pygame.draw.circle(surface, WHITE, (self.x - self.radius // 3, self.y - self.radius // 3), self.radius // 4)

        pygame.draw.circle(surface, DARK_GRAY, (self.x, self.y), self.radius, 2)

    def set_status(self, is_dangerous):
        self.is_red = is_dangerous

class ScrollingImage:

    def __init__(self, image_path, display_rect, target_height=None, is_positive=None):
        try:
            self.original_image = pygame.image.load(image_path)
            self.path = image_path
            self.pil_image = Image.open(image_path)
            self.original_path = image_path

            self.detection_done = False
        except pygame.error as e:
            print(f"Erreur lors du chargement de l'image {image_path}: {e}")
            self.original_image = pygame.Surface((300, 200))
            self.original_image.fill((200, 200, 200))
            pygame.draw.rect(self.original_image, (100, 100, 100), (0, 0, 300, 200), 3)
            self.path = "placeholder"
            self.pil_image = None
            self.original_path = None
            self.detection_done = True

        self.original_width, self.original_height = self.original_image.get_size()

        if target_height:
            target_width = int(self.original_width * (target_height / self.original_height))
            self.image = pygame.transform.scale(self.original_image, (target_width, target_height))
            self.scale_factor_x = target_width / self.original_width
            self.scale_factor_y = target_height / self.original_height
        else:
            self.image = self.original_image
            self.scale_factor_x = 1.0
            self.scale_factor_y = 1.0

        self.width, self.height = self.image.get_size()
        self.x = -self.width
        self.y = display_rect.y + (display_rect.height - self.height) // 2

        self.display_rect = display_rect
        self.is_complete = False
        self.is_positive = is_positive

        self.current_detections = []

    def update(self, speed):
        self.x += speed

        if self.x > WINDOW_WIDTH:
            self.is_complete = True

    def is_in_detection_zone(self, mask_rect):
        image_center = self.x + self.width / 2
        mask_center = mask_rect.x + mask_rect.width / 2
        detection_width = mask_rect.width * 0.2

        return abs(image_center - mask_center) < detection_width

    def get_right_edge(self):
        return self.x + self.width

    def get_left_edge(self):
        return self.x

    def set_detections(self, detections):
        if not detections:
            self.current_detections = []
            self.detection_done = True
            return

        orig_width, orig_height = self.original_width, self.original_height
        disp_width, disp_height = self.width, self.height

        scale_x = disp_width / orig_width
        scale_y = disp_height / orig_height

        self.current_detections = []
        for det in detections:
            detection = det.copy()

            detection['x1_rel'] = int(det['x1'] * scale_x)
            detection['y1_rel'] = int(det['y1'] * scale_y)
            detection['x2_rel'] = int(det['x2'] * scale_x)
            detection['y2_rel'] = int(det['y2'] * scale_y)

            self.current_detections.append(detection)

        self.detection_done = True
        print(f"Détections définies pour {self.path}: {len(self.current_detections)} objets")
        print(f"Dimensions originales: {orig_width}x{orig_height}, Dimensions affichées: {disp_width}x{disp_height}")
        print(f"Facteurs d'échelle: sx={scale_x}, sy={scale_y}")

    def draw_detections(self, surface):
        for det in self.current_detections:
            x1 = self.x + det['x1_rel']
            y1 = self.y + det['y1_rel']
            w = det['x2_rel'] - det['x1_rel']
            h = det['y2_rel'] - det['y1_rel']
            box = pygame.Rect(x1, y1, w, h)
            pygame.draw.rect(surface, det['color'], box, 4)

            r, g, b = det['color']
            text_color = (0, 0, 0) if (r + g + b) / 3 > 200 else (255, 255, 255)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            txt = font_detection.render(label, True, text_color)
            bg = pygame.Rect(x1, y1 - txt.get_height(),
                             txt.get_width() + 4, txt.get_height())
            pygame.draw.rect(surface, det['color'], bg)
            surface.blit(txt, (x1 + 2, y1 - txt.get_height()))

class ImageManager:

    def __init__(self, base_folder, display_rect):
        self.base_folder = base_folder
        self.display_rect = display_rect
        self.image_queue = deque()
        self.active_images = []
        self.target_height = int(display_rect.height * IMAGE_HEIGHT_RATIO)
        self.can_add_new_image = True
        self.paused = False

        self.detector = ObjectDetector()
        self.current_detections = []

    def _find_random_image(self, folder_type, is_positive):
        # Trouve une image aléatoire dans le dossier spécifié
        if folder_type == "train":
            folder = "train"
        elif folder_type == "test":
            folder = "test"
        else:  # Aléatoire entre train et test
            folder = random.choice(["train", "test"])

        if is_positive is None:  # Aléatoire entre pos et neg
            subfolder = random.choice(["pos", "neg"])
        elif is_positive:
            subfolder = "pos"
        else:
            subfolder = "neg"

        path = os.path.join(self.base_folder, "images", folder, subfolder)

        if not os.path.exists(path):
            print(f"Erreur: Le dossier {path} n'existe pas.")
            return None

        image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print(f"Erreur: Aucune image trouvée dans {path}.")
            return None

        # Sélectionner une image aléatoire
        image_file = random.choice(image_files)
        image_path = os.path.join(path, image_file)

        return image_path

    def queue_image(self, folder_type="random", is_positive=None):
        # Ajoute une image à la file d'attente
        image_path = self._find_random_image(folder_type, is_positive)

        if image_path:
            print(f"Image ajoutée à la file d'attente: {image_path}")
            self.image_queue.append((image_path, is_positive))

            # Si aucune image n'est actuellement affichée, charger immédiatement
            if not self.active_images:
                self._load_next_image()
            else:
                # Sinon vérifier si on peut ajouter une nouvelle image
                self._check_load_next_image()

    def _check_load_next_image(self):
        if not self.image_queue:
            return

        if not self.active_images:
            self._load_next_image()
            return

        last_image = self.active_images[-1]

        if last_image.get_left_edge() > MIN_IMAGE_SPACING:
            self._load_next_image()

    def _load_next_image(self):
        if not self.image_queue:
            return

        image_path, is_positive = self.image_queue.popleft()
        new_image = ScrollingImage(image_path, self.display_rect, self.target_height, is_positive)
        self.active_images.append(new_image)

        if new_image.pil_image and new_image.original_path:
            self._detect_single_image(new_image)

    def _detect_single_image(self, image):
        if not image.pil_image or not image.original_path or image.detection_done:
            return

        detections = self.detector.detect_for_path(image.pil_image, image.original_path)
        image.set_detections(detections)

        if image.is_in_detection_zone(self.display_rect):
            self.current_detections = detections

    def update(self, speed):
        if self.paused:
            return self._check_detection_zone()

        for image in self.active_images:
            image.update(speed)

        self.active_images = [img for img in self.active_images if not img.is_complete]

        self._check_load_next_image()

        return self._check_detection_zone()

    def _check_detection_zone(self):
        self.current_detections = []
        has_dangerous = False

        for img in self.active_images:
            if img.is_in_detection_zone(self.display_rect):
                if img.current_detections:
                    self.current_detections = img.current_detections
                    has_dangerous = True

        return has_dangerous

    def toggle_pause(self, is_paused):
        self.paused = is_paused
        return self.paused

    def draw(self, surface, scanner_rect):
        old_clip = surface.get_clip()
        surface.set_clip(scanner_rect)

        for img in self.active_images:
            surface.blit(img.image, (img.x, img.y))
            img.draw_detections(surface)

        surface.set_clip(old_clip)

    def has_active_image(self):
        return len(self.active_images) > 0

    def queue_size(self):
        return len(self.image_queue)

    def active_image_count(self):
        return len(self.active_images)

    def get_current_detections(self):
        return self.current_detections


def create_interface(base_folder):
    left_panel_width = BUTTON_WIDTH + 20

    total_column_height = (3 * BUTTON_HEIGHT + 2 * BUTTON_MARGIN) + 25 + OBSERVATION_HEIGHT + 20 + PAUSE_BUTTON_HEIGHT
    column_start_y = (WINDOW_HEIGHT - total_column_height) // 2

    image_x = left_panel_width + 5

    vertical_offset = 30
    image_y = (WINDOW_HEIGHT - CONVEYOR_HEIGHT) // 2 + vertical_offset
    conveyor_display = ConveyorDisplay(image_x, image_y, CONVEYOR_WIDTH, CONVEYOR_HEIGHT)
    image_manager = ImageManager(base_folder, conveyor_display.rect)
    light_radius = 20
    conveyor_center_x = image_x + CONVEYOR_WIDTH / 2
    light_x = conveyor_center_x
    light_y = image_y - light_radius - 10
    status_light = StatusLight(light_x, light_y, light_radius)

    left_button_x = 10
    test_button_start_y = column_start_y + vertical_offset // 2

    test_label_pos = (left_button_x + BUTTON_WIDTH // 2, test_button_start_y - 30)

    def create_image_action(folder, is_positive):
        return lambda: image_manager.queue_image(folder, is_positive)

    buttons = []

    buttons.append(Button(
        left_button_x, test_button_start_y,
        BUTTON_WIDTH, BUTTON_HEIGHT,
        "Image aléatoire",
        action=create_image_action("test", None)
    ))

    buttons.append(Button(
        left_button_x, test_button_start_y + BUTTON_HEIGHT + BUTTON_MARGIN,
        BUTTON_WIDTH, BUTTON_HEIGHT,
        "Image négative",
        action=create_image_action("test", False)
    ))

    buttons.append(Button(
        left_button_x, test_button_start_y + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN),
        BUTTON_WIDTH, BUTTON_HEIGHT,
        "Image positive",
        action=create_image_action("test", True)
    ))

    obs_panel_y = test_button_start_y + 3 * (BUTTON_HEIGHT + BUTTON_MARGIN) + 25
    observation_panel = ObservationPanel(
        left_button_x, obs_panel_y,
        BUTTON_WIDTH, OBSERVATION_HEIGHT
    )

    pause_button_y = obs_panel_y + OBSERVATION_HEIGHT + 20
    pause_button = ToggleButton(
        left_button_x + (BUTTON_WIDTH - PAUSE_BUTTON_WIDTH) // 2, pause_button_y,
        PAUSE_BUTTON_WIDTH, PAUSE_BUTTON_HEIGHT,
        "Reprendre", "Pause",
        action=lambda is_paused: image_manager.toggle_pause(is_paused),
        color=PAUSE_BUTTON_COLOR,
        hover_color=PAUSE_BUTTON_HOVER_COLOR
    )

    return conveyor_display, image_manager, status_light, buttons, None, test_label_pos, pause_button, observation_panel

def main():
    # Chemin vers le dossier de base (doit contenir un sous-dossier 'images')
    base_folder = "."

    conveyor_display, image_manager, status_light, buttons, train_label_pos, test_label_pos, pause_button, observation_panel = create_interface(
        base_folder)

    clock = pygame.time.Clock()
    running = True

    detection_timer = 0

    print("Application démarrée.")
    print(f"Structure de dossiers attendue: {base_folder}/images/[train|test]/[pos|neg]/")

    if YOLO_AVAILABLE:
        print("Module YOLO détecté. Les fonctionnalités de détection sont activées.")
    else:
        print("Module YOLO non détecté. Les fonctionnalités de détection sont désactivées.")

    # Boucle principale
    while running:
        # Mesurer le temps écoulé depuis la dernière image
        dt = clock.get_time() / 1000.0  # Temps en secondes
        detection_timer += dt

        mouse_pos = pygame.mouse.get_pos()

        for button in buttons:
            button.update(mouse_pos)
        pause_button.update(mouse_pos)

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            for button in buttons:
                if button.is_clicked(event):
                    print(f"Bouton '{button.text}' cliqué")
                    if button.action:
                        button.action()

            if pause_button.is_clicked(event):
                print(f"Bouton Pause/Reprendre activé. État: {pause_button.is_active}")

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                observation_panel.click_handler(event.pos)

        is_dangerous = image_manager.update(SCROLL_SPEED)

        status_light.set_status(is_dangerous)

        observation_panel.update_detections(image_manager.get_current_detections())

        screen.fill(LIGHT_BLUE)

        conveyor_display.draw(screen)

        image_manager.draw(screen, conveyor_display.rect)

        status_light.draw(screen)

        if test_label_pos:
            test_label = font_title.render("TEST", True, DARK_GRAY)
            screen.blit(test_label, (test_label_pos[0] - test_label.get_width() // 2, test_label_pos[1]))

        for button in buttons:
            button.draw(screen)

        pause_button.draw(screen)

        observation_panel.draw(screen)

        title = font_title.render("Scanner de bagages - Détection d'Objets Dangereux", True, DARK_GRAY)
        title_x = (WINDOW_WIDTH - title.get_width()) // 2
        title_y = 20
        screen.blit(title, (title_x, title_y))

        queue_info = font_status.render(
            f"Images en attente: {image_manager.queue_size()} | Images visibles: {image_manager.active_image_count()}",
            True, DARK_GRAY)
        screen.blit(queue_info, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()


# Lancer l'application
if __name__ == "__main__":
    main()