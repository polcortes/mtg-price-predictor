import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from pymongo import MongoClient
from random import randint
import threading

from src.model.consumer import AIConsumer
from src.scanner.escaner_cartas import scan_card
from src.ocr.ocr_service import OCRService
from pathlib import Path
import cv2
import io
import matplotlib.pyplot as plt
from datetime import datetime

assets_path = Path(__file__).parent / "assets"

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Magic the Gathering: Price Predictor")
        self.geometry("400x429")
        self.minsize(400, 429)

        self.current_card = None

        # Conexión principal de los servicios (se inicializan al arrancar la ventana y no al importar)
        self.client = MongoClient("localhost", 27017)
        self.collection = self.client["mtg_data"]["card_price"]
        self.consumer = AIConsumer()
        self.ocr_service = OCRService()

        self._build_ui()
        
    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1) 
        main_frame.columnconfigure(1, weight=3) 
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.camera_img = tk.PhotoImage(file=assets_path / "camera.png")

        btn1_label = ttk.Label(left_frame, text="Escanea una carta")
        btn1_label.pack()
        btn1 = ttk.Button(left_frame, image=self.camera_img, command=self.start_card_scanner)
        btn1.pack(fill="x", pady=5)

        separator = ttk.Separator(left_frame, orient='horizontal')
        separator.pack(fill='x', pady=10)

        self.upload_img = tk.PhotoImage(file=assets_path / "upload-simple.png")

        btn2_label = ttk.Label(left_frame, text="Subir una imagen")
        btn2_label.pack()
        btn2 = ttk.Button(left_frame, image=self.upload_img)
        btn2.pack(fill="x", pady=5)

        self.preview_label = ttk.Label(left_frame, text="Preview", relief="sunken")
        self.preview_label.pack(fill="both", expand=True, pady=(10, 0))

        btn2.config(command=self.on_image_selected)

        self.right_frame = ttk.LabelFrame(
            main_frame, 
            text=f"Datos de {'la carta' if self.current_card is None else self.current_card}", 
            padding="10"
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.columnconfigure(1, weight=1)

        ttk.Label(self.right_frame, text="Nombre:").grid(row=0, column=0, sticky="w", pady=5)
        self.entry_nombre = ttk.Entry(self.right_frame)
        self.entry_nombre.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        ttk.Label(self.right_frame, text="Set:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_coleccion = ttk.Entry(self.right_frame)
        self.entry_coleccion.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        self.submit_btn = ttk.Button(self.right_frame, text="Predecir Precio")
        self.submit_btn.grid(row=3, column=1, sticky="e", pady=15)
        self.submit_btn.config(command=self.predict_price)
        
        # Etiqueta para mostrar los resultados de la IA, que faltaba añadir visualmente
        self.result_label = ttk.Label(self.right_frame, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def start_card_scanner(self):
        """Inicia el escáner de cartas en un thread separado."""
        def scan_callback(card_path: str):
            if card_path:
                self.show_image_preview(card_path)
                self.extract_card_from_image(card_path)

        scan_card(master=self, callback=lambda path: scan_callback(path))
    def on_image_selected(self):
        image_path = askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not image_path:
            return

        self.show_image_preview(image_path)
        self.extract_card_from_image(image_path)

    def show_image_preview(self, image_path):
        if not image_path:
            return
        try:
            img = Image.open(image_path)
            img.thumbnail((150, 200), Image.Resampling.LANCZOS)
            preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=preview_photo, text="")
            self.preview_label.image = preview_photo
        except Exception as e:
            self.preview_label.config(text=f"Error: {str(e)}")

    def extract_card_from_image(self, image_path):
        """Extrae nombre de carta desde imagen usando OCR."""
        try:
            # result = self.ocr_service.extract_card_name(image_path)
            from src.ocr.with_ai import process_mtg_card
            result = process_mtg_card(image_path)

            print(result)

            if result:
                card_name = result.get("validated_data", {}).get("card_name")
                set_name = result.get("validated_data", {}).get("set_name")

                self.entry_nombre.delete(0, tk.END)
                self.entry_nombre.insert(0, card_name or "")

                self.entry_coleccion.delete(0, tk.END)
                self.entry_coleccion.insert(0, set_name or "")

                self.selected_card_name = card_name
                self.selected_set_name = set_name
                self.selected_card_uuid = result.get("card_uuid")

                self.result_label.config(
                    text=f"OCR: {card_name}\nCandidatos: {set_name}",
                    foreground="#0066cc"
                )
            else:
                self.result_label.config(text="Error: No se pudo extraer el nombre", foreground="red")

        except Exception as e:
            self.result_label.config(text=f"Error OCR: {str(e)}", foreground="red")

    def predict_price(self):
        """Obtiene carta aleatoria y predice su precio."""
        self.result_label.config(image="")
        try:
            if not hasattr(self, 'selected_card_uuid') or not self.selected_card_uuid:
                self.result_label.config(text="Error: Primero debes escanear o subir una carta válida.", foreground="red")
                return

            # Obtener mapa de precios
            try:
                card = self.collection.find({"cardId": self.selected_card_uuid}).limit(1).next()
            except StopIteration:
                self.result_label.config(text="Error: No hay historial de precios para esta carta específica.", foreground="red")
                return

            prices_map = card.get('datePriceMap', {})
            
            if not prices_map:
                self.result_label.config(text="Error: Carta sin historial de precios", foreground="red")
                return
                
            # Hacer predicción
            probability = self.consumer.predict(prices_map)

            print(f"Probability: {probability * 100}%")
            
            if probability is None:
                self.result_label.config(text=f"Error: Se necesitan {self.consumer.days_back} días de datos", foreground="red")
                return
                
            # Mostrar resultado con color según probabilidad
            percentage = probability * 100
            if percentage > 70:
                color = "#28a745"  # Verde
                trend = "FUERTE SUBIDA 📈"
            elif percentage > 50:
                color = "#ffc107"  # Amarillo
                trend = "SUBIDA LEVE 📊"
            else:
                color = "#dc3545"  # Rojo
                trend = "BAJADA 📉"
                
            result_text = f"Probabilidad de subida: {percentage:.1f}%\nTendencia: {trend}"

            # Generar gráfica con pyplot
            fig, ax = plt.subplots(figsize=(4, 2.5))
            dates = sorted(prices_map.keys())
            try:
                date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
            except ValueError:
                date_objects = dates
            prices = [prices_map[d] for d in dates]
            
            ax.plot(date_objects, prices, color=color, linewidth=2)
            ax.set_title("Evolución del Precio", fontsize=10)
            ax.set_ylabel("€", fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close(fig)

            img = Image.open(buf)
            self.graph_photo = ImageTk.PhotoImage(img)

            self.result_label.config(
                text=result_text, 
                image=self.graph_photo, 
                compound=tk.TOP,
                foreground=color
            )
            
            # Actualizar título del frame derecho
            self.right_frame.config(text=f"Datos de {getattr(self, 'selected_card_name', 'la carta')}")
            
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", image="", foreground="red")
            print(f"Error completo: {e}")