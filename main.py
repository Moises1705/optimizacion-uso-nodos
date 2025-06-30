import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from itertools import permutations
import json
import time

class CircuitoElectronico:
    """Clase para representar un circuito electrónico como grafo"""
    
    def __init__(self, nombre="Circuito"):
        self.nombre = nombre    
        self.grafo = nx.Graph()
        self.componentes = {}  # {nodo: tipo_componente}
        self.conexiones = {}   # {arista: resistencia/impedancia}
        
    def agregar_componente(self, id_componente, tipo, posicion=None):
        """Agregar un componente al circuito"""
        self.grafo.add_node(id_componente)
        self.componentes[id_componente] = tipo
        if posicion:
            self.grafo.nodes[id_componente]['pos'] = posicion
            
    def agregar_conexion(self, comp1, comp2, resistencia=1.0):
        """Agregar conexión entre componentes"""
        self.grafo.add_edge(comp1, comp2)
        self.conexiones[(comp1, comp2)] = resistencia
        
    def obtener_matriz_adyacencia(self):
        """Obtener matriz de adyacencia del circuito"""
        return nx.adjacency_matrix(self.grafo).todense()

class AnalizadorHamiltoniano:
    """Clase para análisis de caminos hamiltonianos en circuitos"""
    
    def __init__(self, circuito):
        self.circuito = circuito
        
    def es_hamiltoniano(self):
        """Verificar si existe un camino hamiltoniano"""
        try:
            camino = self.encontrar_camino_hamiltoniano()
            return camino is not None
        except:
            return False
            
    def encontrar_camino_hamiltoniano(self):
        """Encontrar camino hamiltoniano usando backtracking"""
        nodos = list(self.circuito.grafo.nodes())
        if len(nodos) < 2:
            return nodos
            
        def backtrack(camino, visitados):
            if len(camino) == len(nodos):
                return camino
                
            ultimo_nodo = camino[-1]
            for vecino in self.circuito.grafo.neighbors(ultimo_nodo):
                if vecino not in visitados:
                    nuevo_camino = camino + [vecino]
                    nuevos_visitados = visitados | {vecino}
                    resultado = backtrack(nuevo_camino, nuevos_visitados)
                    if resultado:
                        return resultado
            return None
            
        # Probar desde cada nodo como punto de inicio
        for inicio in nodos:
            camino = backtrack([inicio], {inicio})
            if camino:
                return camino
        return None
        
    def encontrar_ciclo_hamiltoniano(self):
        """Encontrar ciclo hamiltoniano si existe"""
        camino = self.encontrar_camino_hamiltoniano()
        if not camino:
            return None
            
        # Verificar si se puede cerrar el ciclo
        if self.circuito.grafo.has_edge(camino[-1], camino[0]):
            return camino + [camino[0]]
        return None
        
    def calcular_eficiencia_energetica(self, camino):
        """Calcular eficiencia energética del camino"""
        if not camino or len(camino) < 2:
            return 0
            
        resistencia_total = 0
        for i in range(len(camino) - 1):
            arista = (camino[i], camino[i+1])
            if arista in self.circuito.conexiones:
                resistencia_total += self.circuito.conexiones[arista]
            elif (arista[1], arista[0]) in self.circuito.conexiones:
                resistencia_total += self.circuito.conexiones[(arista[1], arista[0])]
            else:
                resistencia_total += 1.0  # Resistencia por defecto
                
        # Eficiencia inversamente proporcional a la resistencia
        return 1.0 / (1.0 + resistencia_total)

class VisualizadorCircuito:
    """Clase para visualizar circuitos y caminos hamiltonianos"""
    
    def __init__(self, parent):
        self.parent = parent
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        
    def dibujar_circuito(self, circuito, camino_hamiltoniano=None):
        """Dibujar el circuito con opción de resaltar camino hamiltoniano"""
        self.ax.clear()
        
        if len(circuito.grafo.nodes()) == 0:
            self.ax.text(0.5, 0.5, 'No hay componentes en el circuito', 
                        horizontalalignment='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
            
        # Generar posiciones si no existen
        pos = {}
        for nodo in circuito.grafo.nodes():
            if 'pos' in circuito.grafo.nodes[nodo]:
                pos[nodo] = circuito.grafo.nodes[nodo]['pos']
        
        if not pos:
            pos = nx.spring_layout(circuito.grafo, seed=42)
            
        # Dibujar todas las aristas en gris claro
        nx.draw_networkx_edges(circuito.grafo, pos, edge_color='lightgray', 
                              width=1, alpha=0.6, ax=self.ax)
        
        # Resaltar camino hamiltoniano si existe
        if camino_hamiltoniano:
            camino_aristas = [(camino_hamiltoniano[i], camino_hamiltoniano[i+1]) 
                             for i in range(len(camino_hamiltoniano)-1)]
            nx.draw_networkx_edges(circuito.grafo, pos, edgelist=camino_aristas,
                                 edge_color='red', width=3, ax=self.ax)
        
        # Dibujar nodos con colores según tipo de componente
        colores_componentes = {
            'Resistor': 'lightblue',
            'Capacitor': 'lightgreen', 
            'Inductor': 'lightyellow',
            'Transistor': 'lightcoral',
            'IC': 'lightpink',
            'Fuente': 'gold'
        }
        
        for tipo in set(circuito.componentes.values()):
            nodos_tipo = [n for n, t in circuito.componentes.items() if t == tipo]
            if nodos_tipo:
                nx.draw_networkx_nodes(circuito.grafo, pos, nodelist=nodos_tipo,
                                     node_color=colores_componentes.get(tipo, 'white'),
                                     node_size=800, ax=self.ax)
        
        # Etiquetas de nodos
        nx.draw_networkx_labels(circuito.grafo, pos, font_size=8, ax=self.ax)
        
        self.ax.set_title(f'Circuito: {circuito.nombre}')
        self.ax.axis('off')
        self.canvas.draw()
        
    def get_widget(self):
        return self.canvas.get_tk_widget()

class AplicacionHamiltonianoCircuitos:
    """Aplicación principal para análisis de grafos hamiltonianos en circuitos"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Grafos Hamiltonianos en Circuitos Electrónicos")
        self.root.geometry("1400x900")
        
        self.circuito_actual = CircuitoElectronico()
        self.analizador = None
        
        self.setup_ui()
        self.crear_circuito_ejemplo()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Sección de componentes
        comp_frame = ttk.LabelFrame(control_frame, text="Gestión de Componentes")
        comp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Agregar componente
        ttk.Label(comp_frame, text="ID Componente:").pack(anchor=tk.W)
        self.id_entry = ttk.Entry(comp_frame)
        self.id_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(comp_frame, text="Tipo:").pack(anchor=tk.W)
        self.tipo_combo = ttk.Combobox(comp_frame, values=[
            "Resistor", "Capacitor", "Inductor", "Transistor", "IC", "Fuente"
        ])
        self.tipo_combo.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(comp_frame, text="Agregar Componente", 
                  command=self.agregar_componente).pack(pady=5)
        
        # Sección de conexiones
        conn_frame = ttk.LabelFrame(control_frame, text="Gestión de Conexiones")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Componente 1:").pack(anchor=tk.W)
        self.comp1_entry = ttk.Entry(conn_frame)
        self.comp1_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(conn_frame, text="Componente 2:").pack(anchor=tk.W)
        self.comp2_entry = ttk.Entry(conn_frame)
        self.comp2_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(conn_frame, text="Resistencia (Ω):").pack(anchor=tk.W)
        self.resistencia_entry = ttk.Entry(conn_frame)
        self.resistencia_entry.insert(0, "1.0")
        self.resistencia_entry.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(conn_frame, text="Agregar Conexión", 
                  command=self.agregar_conexion).pack(pady=5)
        
        # Sección de análisis
        analisis_frame = ttk.LabelFrame(control_frame, text="Análisis Hamiltoniano")
        analisis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(analisis_frame, text="Encontrar Camino Hamiltoniano", 
                  command=self.analizar_hamiltoniano).pack(fill=tk.X, pady=2)
        
        ttk.Button(analisis_frame, text="Encontrar Ciclo Hamiltoniano", 
                  command=self.analizar_ciclo_hamiltoniano).pack(fill=tk.X, pady=2)
        
        ttk.Button(analisis_frame, text="Calcular Eficiencia Energética", 
                  command=self.calcular_eficiencia).pack(fill=tk.X, pady=2)
        
        # Botones de archivo
        archivo_frame = ttk.LabelFrame(control_frame, text="Archivo")
        archivo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(archivo_frame, text="Nuevo Circuito", 
                  command=self.nuevo_circuito).pack(fill=tk.X, pady=2)
        
        ttk.Button(archivo_frame, text="Cargar Ejemplo", 
                  command=self.crear_circuito_ejemplo).pack(fill=tk.X, pady=2)
        
        # Panel derecho - Visualización y resultados
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Visualizador
        self.visualizador = VisualizadorCircuito(right_frame)
        self.visualizador.get_widget().pack(fill=tk.BOTH, expand=True)
        
        # Área de resultados
        resultados_frame = ttk.LabelFrame(right_frame, text="Resultados")
        resultados_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.resultado_text = tk.Text(resultados_frame, height=8)
        scrollbar = ttk.Scrollbar(resultados_frame, orient=tk.VERTICAL, 
                                 command=self.resultado_text.yview)
        self.resultado_text.configure(yscrollcommand=scrollbar.set)
        
        self.resultado_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def agregar_componente(self):
        """Agregar un nuevo componente al circuito"""
        id_comp = self.id_entry.get().strip()
        tipo = self.tipo_combo.get()
        
        if not id_comp or not tipo:
            messagebox.showerror("Error", "Debe especificar ID y tipo del componente")
            return
            
        if id_comp in self.circuito_actual.componentes:
            messagebox.showerror("Error", f"El componente {id_comp} ya existe")
            return
            
        self.circuito_actual.agregar_componente(id_comp, tipo)
        self.actualizar_visualizacion()
        self.mostrar_resultado(f"Componente {id_comp} ({tipo}) agregado exitosamente")
        
        # Limpiar campos
        self.id_entry.delete(0, tk.END)
        self.tipo_combo.set("")
        
    def agregar_conexion(self):
        """Agregar una nueva conexión entre componentes"""
        comp1 = self.comp1_entry.get().strip()
        comp2 = self.comp2_entry.get().strip()
        
        try:
            resistencia = float(self.resistencia_entry.get())
        except ValueError:
            messagebox.showerror("Error", "La resistencia debe ser un número válido")
            return
            
        if not comp1 or not comp2:
            messagebox.showerror("Error", "Debe especificar ambos componentes")
            return
            
        if comp1 not in self.circuito_actual.componentes or comp2 not in self.circuito_actual.componentes:
            messagebox.showerror("Error", "Ambos componentes deben existir en el circuito")
            return
            
        if comp1 == comp2:
            messagebox.showerror("Error", "No se puede conectar un componente consigo mismo")
            return
            
        self.circuito_actual.agregar_conexion(comp1, comp2, resistencia)
        self.actualizar_visualizacion()
        self.mostrar_resultado(f"Conexión agregada entre {comp1} y {comp2} (R={resistencia}Ω)")
        
        # Limpiar campos
        self.comp1_entry.delete(0, tk.END)
        self.comp2_entry.delete(0, tk.END)
        self.resistencia_entry.delete(0, tk.END)
        self.resistencia_entry.insert(0, "1.0")
        
    def analizar_hamiltoniano(self):
        """Analizar y encontrar camino hamiltoniano"""
        if len(self.circuito_actual.grafo.nodes()) < 2:
            messagebox.showwarning("Advertencia", "El circuito debe tener al menos 2 componentes")
            return
            
        self.analizador = AnalizadorHamiltoniano(self.circuito_actual)
        
        inicio = time.time()
        camino = self.analizador.encontrar_camino_hamiltoniano()
        tiempo_calculo = time.time() - inicio
        
        if camino:
            self.mostrar_resultado(f"✓ CAMINO HAMILTONIANO ENCONTRADO")
            self.mostrar_resultado(f"Camino: {' → '.join(camino)}")
            self.mostrar_resultado(f"Longitud: {len(camino)} componentes")
            self.mostrar_resultado(f"Tiempo de cálculo: {tiempo_calculo:.4f} segundos")
            
            # Calcular eficiencia
            eficiencia = self.analizador.calcular_eficiencia_energetica(camino)
            self.mostrar_resultado(f"Eficiencia energética: {eficiencia:.4f}")
            
            # Visualizar con camino resaltado
            self.visualizador.dibujar_circuito(self.circuito_actual, camino)
        else:
            self.mostrar_resultado("✗ NO EXISTE CAMINO HAMILTONIANO")
            self.mostrar_resultado("El circuito no puede ser recorrido visitando cada componente exactamente una vez")
            self.actualizar_visualizacion()
            
        self.analizar_propiedades_circuito()
        
    def analizar_ciclo_hamiltoniano(self):
        """Analizar y encontrar ciclo hamiltoniano"""
        if len(self.circuito_actual.grafo.nodes()) < 3:
            messagebox.showwarning("Advertencia", "El circuito debe tener al menos 3 componentes para un ciclo")
            return
            
        self.analizador = AnalizadorHamiltoniano(self.circuito_actual)
        
        inicio = time.time()
        ciclo = self.analizador.encontrar_ciclo_hamiltoniano()
        tiempo_calculo = time.time() - inicio
        
        if ciclo:
            self.mostrar_resultado(f"✓ CICLO HAMILTONIANO ENCONTRADO")
            self.mostrar_resultado(f"Ciclo: {' → '.join(ciclo)}")
            self.mostrar_resultado(f"Longitud: {len(ciclo)-1} componentes únicos")
            self.mostrar_resultado(f"Tiempo de cálculo: {tiempo_calculo:.4f} segundos")
            
            # Calcular eficiencia
            eficiencia = self.analizador.calcular_eficiencia_energetica(ciclo)
            self.mostrar_resultado(f"Eficiencia energética: {eficiencia:.4f}")
            
            # Visualizar con ciclo resaltado
            self.visualizador.dibujar_circuito(self.circuito_actual, ciclo)
        else:
            self.mostrar_resultado("✗ NO EXISTE CICLO HAMILTONIANO")
            self.mostrar_resultado("El circuito no puede formar un ciclo visitando cada componente exactamente una vez")
            self.actualizar_visualizacion()
            
    def calcular_eficiencia(self):
        """Calcular y mostrar análisis de eficiencia energética"""
        if not self.analizador:
            self.analizador = AnalizadorHamiltoniano(self.circuito_actual)
            
        camino = self.analizador.encontrar_camino_hamiltoniano()
        if not camino:
            messagebox.showinfo("Información", "Primero debe encontrar un camino hamiltoniano")
            return
            
        eficiencia = self.analizador.calcular_eficiencia_energetica(camino)
        
        # Calcular resistencia total
        resistencia_total = 0
        for i in range(len(camino) - 1):
            arista = (camino[i], camino[i+1])
            if arista in self.circuito_actual.conexiones:
                resistencia_total += self.circuito_actual.conexiones[arista]
            elif (arista[1], arista[0]) in self.circuito_actual.conexiones:
                resistencia_total += self.circuito_actual.conexiones[(arista[1], arista[0])]
            else:
                resistencia_total += 1.0
                
        self.mostrar_resultado("=== ANÁLISIS DE EFICIENCIA ENERGÉTICA ===")
        self.mostrar_resultado(f"Resistencia total del camino: {resistencia_total:.2f} Ω")
        self.mostrar_resultado(f"Eficiencia energética: {eficiencia:.4f}")
        self.mostrar_resultado(f"Pérdida de potencia relativa: {1-eficiencia:.4f}")
        
        if eficiencia > 0.7:
            self.mostrar_resultado("✓ Circuito con ALTA eficiencia energética")
        elif eficiencia > 0.4:
            self.mostrar_resultado("⚠ Circuito con eficiencia energética MEDIA")
        else:
            self.mostrar_resultado("✗ Circuito con BAJA eficiencia energética")
            
    def analizar_propiedades_circuito(self):
        """Analizar propiedades del circuito"""
        grafo = self.circuito_actual.grafo
        
        self.mostrar_resultado("\n=== PROPIEDADES DEL CIRCUITO ===")
        self.mostrar_resultado(f"Número de componentes: {len(grafo.nodes())}")
        self.mostrar_resultado(f"Número de conexiones: {len(grafo.edges())}")
        self.mostrar_resultado(f"Densidad del grafo: {nx.density(grafo):.3f}")
        
        if len(grafo.nodes()) > 0:
            conectado = nx.is_connected(grafo)
            self.mostrar_resultado(f"Circuito conectado: {'Sí' if conectado else 'No'}")
            
            if conectado:
                diametro = nx.diameter(grafo)
                self.mostrar_resultado(f"Diámetro del circuito: {diametro}")
                
        # Análisis de grados
        grados = dict(grafo.degree())
        if grados:
            grado_promedio = sum(grados.values()) / len(grados)
            self.mostrar_resultado(f"Grado promedio: {grado_promedio:.2f}")
            
    def crear_circuito_ejemplo(self):
        """Crear un circuito de ejemplo"""
        self.nuevo_circuito()
        
        # Agregar componentes de ejemplo
        componentes_ejemplo = [
            ("R1", "Resistor"), ("R2", "Resistor"), ("C1", "Capacitor"),
            ("L1", "Inductor"), ("T1", "Transistor"), ("VCC", "Fuente")
        ]
        
        for id_comp, tipo in componentes_ejemplo:
            self.circuito_actual.agregar_componente(id_comp, tipo)
            
        # Agregar conexiones de ejemplo
        conexiones_ejemplo = [
            ("VCC", "R1", 10.0), ("R1", "C1", 5.0), ("C1", "L1", 3.0),
            ("L1", "T1", 8.0), ("T1", "R2", 6.0), ("R2", "VCC", 4.0)
        ]
        
        for comp1, comp2, resistencia in conexiones_ejemplo:
            self.circuito_actual.agregar_conexion(comp1, comp2, resistencia)
            
        self.actualizar_visualizacion()
        self.mostrar_resultado("Circuito de ejemplo cargado exitosamente")
        self.mostrar_resultado("Este circuito tiene un ciclo hamiltoniano")
        
    def nuevo_circuito(self):
        """Crear un nuevo circuito vacío"""
        self.circuito_actual = CircuitoElectronico("Nuevo Circuito")
        self.analizador = None
        self.actualizar_visualizacion()
        self.limpiar_resultados()
        
    def actualizar_visualizacion(self):
        """Actualizar la visualización del circuito"""
        self.visualizador.dibujar_circuito(self.circuito_actual)
        
    def mostrar_resultado(self, texto):
        """Mostrar resultado en el área de texto"""
        self.resultado_text.insert(tk.END, texto + "\n")
        self.resultado_text.see(tk.END)
        
    def limpiar_resultados(self):
        """Limpiar el área de resultados"""
        self.resultado_text.delete(1.0, tk.END)

def main():
    """Función principal para ejecutar la aplicación"""
    root = tk.Tk()
    app = AplicacionHamiltonianoCircuitos(root)
    
    # Mensaje de bienvenida
    app.mostrar_resultado("=== APLICACIÓN DE GRAFOS HAMILTONIANOS EN CIRCUITOS ELECTRÓNICOS ===")
    app.mostrar_resultado("Esta aplicación permite:")
    app.mostrar_resultado("• Diseñar circuitos electrónicos como grafos")
    app.mostrar_resultado("• Encontrar caminos y ciclos hamiltonianos")
    app.mostrar_resultado("• Analizar eficiencia energética")
    app.mostrar_resultado("• Optimizar el diseño de circuitos integrados")
    app.mostrar_resultado("\nUse los controles de la izquierda para construir su circuito.")
    app.mostrar_resultado("¡Circuito de ejemplo cargado para comenzar!")
    
    root.mainloop()

if __name__ == "__main__":
    main()