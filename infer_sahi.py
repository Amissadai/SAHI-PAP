import cv2
import numpy as np
from ultralytics import YOLO
from enviar_img_servidor import configuracao_servidor, criar_bucket, download_dado_minio

def load_model(model_path):
    """Carrega o modelo YOLO a partir do caminho especificado"""
    return YOLO(model_path)

def load_and_preprocess_image(image_path, target_size):
    """Carrega e redimensiona a imagem para o tamanho especificado"""
    image = cv2.imread(image_path)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

def process_single_tile(tile, model, conf_threshold):
    """Processa um único tile e retorna detecções normalizadas e tile anotado"""
    boxes_normalized = []
    scores = []
    classes = []
    annotated_tile = None
    results = model(tile, verbose=False)
    
    for result in results:
        # Extrai informações das detecções
        boxes = result.boxes.xyxyn.cpu().numpy()
        classes_result = result.boxes.cls.cpu().numpy()
        scores_result = result.boxes.conf.cpu().numpy()
        
        # Filtra detecções por confiança
        for box, cls, score in zip(boxes, classes_result, scores_result):
            if score > conf_threshold:
                boxes_normalized.append(box)
                scores.append(score)
                classes.append(cls)
        
        # Gera visualização se houver detecções
        if len(boxes) > 0:
            annotated_tile = result.plot()
    
    return boxes_normalized, scores, classes, annotated_tile

def process_tiles(image, model, tile_size, step, conf_threshold):
    """Processa a imagem em tiles e retorna todas as detecções"""
    height, width = image.shape[:2]
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for y in range(0, height - tile_size[1] + 1, step):
        for x in range(0, width - tile_size[0] + 1, step):
            x2 = x + tile_size[0]
            y2 = y + tile_size[1]
            tile = image[y:y2, x:x2]
            
            # Processa o tile atual
            boxes_norm, scores, classes, annotated_tile = process_single_tile(
                tile, model, conf_threshold
            )
            
            # Converte coordenadas normalizadas para absolutas
            for box, score, cls in zip(boxes_norm, scores, classes):
                x1_t, y1_t, x2_t, y2_t = box
                x1 = int(x1_t * tile_size[0]) + x
                y1 = int(y1_t * tile_size[1]) + y
                x2 = int(x2_t * tile_size[0]) + x
                y2 = int(y2_t * tile_size[1]) + y
                
                all_boxes.append((x1, y1, x2, y2))
                all_scores.append(score)
                all_classes.append(cls)
            
            # Mostra tile processado
            if annotated_tile is not None:
                cv2.imshow('Processando Tiles...', annotated_tile)
                if cv2.waitKey(1) == 27:  # ESC para interromper
                    break
                
    return all_boxes, all_scores, all_classes

def convert_boxes_to_nms_format(boxes):
    """Converte caixas do formato (x1,y1,x2,y2) para (x,y,largura,altura)"""
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]

def apply_nms(boxes, scores, conf_threshold, nms_threshold):
    """Aplica Non-Maximum Suppression nas detecções"""
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    return indices.flatten() if len(indices) > 0 else []

def draw_detections(image, boxes, scores, classes, model):
    """Desenha as detecções finais na imagem"""
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        label = f"{model.names[int(cls)]} {score:.2f}"
       
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
        
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
    return image

def main():

    arquivo_remoto = "dataset/04927.bmp"
    minio_client, bucket_name = configuracao_servidor()
    criar_bucket(minio_client, bucket_name)
    download_dado_minio(minio_client, bucket_name, arquivo_remoto, "dataset_infer/result_infer.png")

    # Configurações
    CONFIG = {
        'model_path': 'best.pt',
        'image_path': 'dataset_infer/result_infer.png',
        'target_size': (1280, 640),
        'tile_size': (320, 320),
        'step': 10,
        'nms_threshold': 0.1,
        'conf_threshold': 0.9
    }

    # Inicialização do modelo
    model = load_model(CONFIG['model_path'])
    
    # Carregar imagem
    image = load_and_preprocess_image(
        CONFIG['image_path'], CONFIG['target_size']
    )
    
    # Processamento por tiles
    all_boxes, all_scores, all_classes = process_tiles(
        image=image,
        model=model,
        tile_size=CONFIG['tile_size'],
        step=CONFIG['step'],
        conf_threshold=CONFIG['conf_threshold']
    )
    
    # Conversão para formato do NMS
    nms_boxes = convert_boxes_to_nms_format(all_boxes)
    
    # Aplicar NMS
    indices = apply_nms(
        boxes=nms_boxes,
        scores=all_scores,
        conf_threshold=CONFIG['conf_threshold'],
        nms_threshold=CONFIG['nms_threshold']
    )
    
    # Filtrar detecções finais
    if len(indices) > 0:
        final_boxes = [all_boxes[i] for i in indices]
        final_scores = [all_scores[i] for i in indices]
        final_classes = [all_classes[i] for i in indices]
    else:
        final_boxes, final_scores, final_classes = [], [], []
    
    # Desenhar resultados
    result_image = draw_detections(
        image.copy(), final_boxes, final_scores, final_classes, model
    )
    

    cv2.imshow('Resultado Final - YOLOv8', result_image)
    cv2.waitKey(0)
    cv2.imwrite('resultado_final_nms.jpg', result_image)
   

if __name__ == "__main__":
    main()