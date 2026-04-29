import tkinter as tk
from tkinter import ttk

class LoadingModelWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Cargando Modelo")
        # Establecemos un tamaño adecuado para los mensajes
        self.geometry("400x150")
        self.resizable(False, False)
        
        # Deshabilitamos el botón de cerrar (la "X") para forzar a que sea el código quien lo cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self._build_ui()
        
    def _build_ui(self):
        container = ttk.Frame(self, padding="20")
        container.pack(fill=tk.BOTH, expand=True)
        
        label_title = ttk.Label(
            container, 
            text="Compilando el modelo LLM...", 
            font=("Helvetica", 12, "bold")
        )
        label_title.pack(pady=(0, 10))
        
        label_subtitle = ttk.Label(
            container, 
            text="Por favor, no cierre el programa. Esto puede tomar unos minutos.", 
            font=("Helvetica", 10)
        )
        label_subtitle.pack(pady=(0, 15))
        
        # Una barra de progreso o un indicador de actividad
        self.progress = ttk.Progressbar(container, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        self.progress.start(15)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - self.winfo_reqwidth()) // 2
        y = (screen_height - self.winfo_reqheight()) // 2
        self.geometry(f"+{x}+{y}")
        
    def on_closing(self):
        """
        Al intentar cerrar la ventana con la X, no hacemos nada.
        """
        pass
        
    def close_window(self):
        """
        Método para ser llamado desde código cuando el modelo termine de cargar.
        """
        self.progress.stop()
        self.quit()
        self.destroy()