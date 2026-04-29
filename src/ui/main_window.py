import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from pymongo import MongoClient
from random import randint

from src.model.consumer import AIConsumer
from pathlib import Path

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
        btn1 = ttk.Button(left_frame, image=self.camera_img)
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

        btn2.config(command=lambda: self.show_image_preview(
            askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        ))

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

        ttk.Label(self.right_frame, text="Colección:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_coleccion = ttk.Entry(self.right_frame)
        self.entry_coleccion.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        ttk.Label(self.right_frame, text="Estado:").grid(row=2, column=0, sticky="w", pady=5)
        self.entry_estado = ttk.Entry(self.right_frame)
        self.entry_estado.grid(row=2, column=1, sticky="ew", pady=5, padx=5)

        self.submit_btn = ttk.Button(self.right_frame, text="Predecir Precio")
        self.submit_btn.grid(row=3, column=1, sticky="e", pady=15)
        self.submit_btn.config(command=self.predict_price)
        
        # Etiqueta para mostrar los resultados de la IA, que faltaba añadir visualmente
        self.result_label = ttk.Label(self.right_frame, text="")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

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

    def predict_price(self):
        """Obtiene carta aleatoria y predice su precio."""
        try:
            count = self.collection.count_documents({})
            if count == 0:
                self.result_label.config(text="Error: No hay cartas en la BD", foreground="red")
                return
                
            random_card = self.collection.find().limit(-1).skip(randint(0, count - 1)).next()
            self.current_card = random_card.get('name', 'Desconocida')
            
            # Actualizar UI con datos de la carta
            self.entry_nombre.delete(0, tk.END)
            self.entry_nombre.insert(0, self.current_card)
            
            self.entry_coleccion.delete(0, tk.END)
            self.entry_coleccion.insert(0, random_card.get('set', 'Desconocida'))
            
            # Obtener mapa de precios
            prices_map = random_card.get('datePriceMap', {})
            
            if not prices_map:
                self.result_label.config(text="Error: Carta sin historial de precios", foreground="red")
                return
                
            # Hacer predicción
            probability = self.consumer.predict(prices_map)
            
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
            self.result_label.config(text=result_text, foreground=color)
            
            # Actualizar título del frame derecho
            self.right_frame.config(text=f"Datos de {self.current_card}")
            
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", foreground="red")
            print(f"Error completo: {e}")