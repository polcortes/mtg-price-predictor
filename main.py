import threading
import tkinter as tk
from pathlib import Path

def main():
    base_path = Path(__file__).parent
    model_path = base_path / "model" / "model.keras"
    scaler_path = base_path / "model" / "scaler.pkl"
    config_path = base_path / "model" / "config.pkl"
    
    if not model_path.exists() or not scaler_path.exists() or not config_path.exists():
        from src.ui.loading_model_window import LoadingModelWindow
        
        loading_window = LoadingModelWindow()
        
        def compile_task():
            from src.model.compile_model import compile_model
            try:
                compile_model()
            finally:
                loading_window.after(0, loading_window.close_window)
                
        thread = threading.Thread(target=compile_task, daemon=True)
        thread.start()
        
        # Esto bloquea la ejecución para mostrar la ventana de carga hasta que se llame su .quit()
        loading_window.mainloop()
        
    # Solo cuando existe el modelo (y/o se haya compilado) importamos MainWindow
    from src.ui import MainWindow
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()