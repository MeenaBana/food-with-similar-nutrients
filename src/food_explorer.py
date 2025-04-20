"""
Interactive Food Explorer.
A dashboard for exploring and finding similar food items.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import traceback
from pathlib import Path
from PIL import Image, ImageTk
import json
import webbrowser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_explorer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FoodExplorerApp:
    """Interactive application for exploring food nutrient data."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Food Nutrient Explorer")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
       
        if not self._check_data_exists():
            self._setup_no_data_view()
            return
        
        self._load_data()
        self._setup_ui()
    
    def _check_data_exists(self):
        """Check if necessary data files exist."""
        required_files = [
            'data/processed/food_names.csv',
            'data/results/cluster_labels.npy',
            'data/results/umap_embedding.npy'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                return False
        
        return True
    
    def _setup_no_data_view(self):
        """Show a message and button to run the analysis first."""
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            frame, 
            text="Food analysis data not found. Please run the analysis first.",
            font=("Arial", 14)
        ).pack(pady=20)
        
        ttk.Button(
            frame, 
            text="Run Analysis", 
            command=self._run_analysis
        ).pack(pady=10)
    
    def _run_analysis(self):
        """Run the food analysis script."""
        try:
            parent_dir = str(Path(__file__).resolve().parent)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
           
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Running Analysis")
            progress_window.geometry("400x150")
            ttk.Label(
                progress_window, 
                text="Running food analysis pipeline...\nThis may take a few minutes.",
                font=("Arial", 12)
            ).pack(pady=20)
            
            progress = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="indeterminate")
            progress.pack(pady=10)
            progress.start()
            
            progress_window.update()
           
            import subprocess
            
            analysis_script = Path('src/food_analysis.py')
            if not analysis_script.exists():
                messagebox.showerror("Error", "Analysis script not found: src/food_analysis.py")
                progress_window.destroy()
                return
            
            result = subprocess.run(['python', str(analysis_script)], 
                                   capture_output=True, text=True, check=False)
            
            progress.stop()
            progress_window.destroy()
            
            if result.returncode == 0:
                messagebox.showinfo(
                    "Analysis Complete",
                    "Analysis completed successfully!"
                )
                
                self.root.destroy()
                new_root = tk.Tk()
                FoodExplorerApp(new_root)
                new_root.mainloop()
            else:
                messagebox.showerror(
                    "Analysis Failed",
                    f"Error running analysis:\n{result.stderr}"
                )
            
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to run analysis: {str(e)}")
    
    def _load_data(self):
        """Load the data files."""
        try:
            self.food_names = pd.read_csv('data/processed/food_names.csv', header=None).iloc[:, 0]
            self.labels = np.load('data/results/cluster_labels.npy')
            self.embedding = np.load('data/results/umap_embedding.npy')
            
            min_length = min(len(self.food_names), len(self.labels), self.embedding.shape[0])
            self.food_names = self.food_names[:min_length]
            self.labels = self.labels[:min_length]
            self.embedding = self.embedding[:min_length]
            
            try:
                self.summary = json.load(open('data/results/summary.json'))[0]
            except:
                self.summary = {"n_clusters": len(set(self.labels))}
                
            try:
                self.cluster_data = pd.read_csv('data/results/cluster_data.csv')
            except:
                try:
                    self.cluster_summary = pd.read_csv('data/results/cluster_summary.csv')
                except:
                    self.cluster_summary = None
            
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            sys.exit(1)
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
       
        ttk.Label(
            self.main_frame, 
            text="Norwegian Food Nutrient Explorer",
            font=("Arial", 16, "bold")
        ).pack(pady=10)
       
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
      
        self.similar_foods_tab = ttk.Frame(self.notebook, padding=10)
        self.cluster_explorer_tab = ttk.Frame(self.notebook, padding=10)
        self.visualizations_tab = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.similar_foods_tab, text="Find Similar Foods")
        self.notebook.add(self.cluster_explorer_tab, text="Cluster Explorer")
        self.notebook.add(self.visualizations_tab, text="Visualizations")
       
        self._setup_similar_foods_tab()
        self._setup_cluster_explorer_tab()
        self._setup_visualizations_tab()
       
        self.status_var = tk.StringVar()
        self.status_var.set(f"Loaded {len(self.food_names)} foods in {self.summary['n_clusters']} clusters")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_similar_foods_tab(self):
        """Set up the 'Find Similar Foods' tab."""
        left_frame = ttk.Frame(self.similar_foods_tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
       
        search_frame = ttk.LabelFrame(left_frame, text="Search Food", padding=10)
        search_frame.pack(fill=tk.X, pady=5)
      
        ttk.Label(search_frame, text="Enter food name:").pack(anchor=tk.W)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        self.search_entry.pack(fill=tk.X, pady=5)
        self.search_entry.bind("<KeyRelease>", self._update_suggestions)
       
        self.suggestions_frame = ttk.Frame(search_frame)
        self.suggestions_frame.pack(fill=tk.X)
        
        self.suggestions_listbox = tk.Listbox(self.suggestions_frame, height=5)
        self.suggestions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.suggestions_listbox.bind("<<ListboxSelect>>", self._select_suggestion)
        
        suggestions_scrollbar = ttk.Scrollbar(self.suggestions_frame, orient=tk.VERTICAL, command=self.suggestions_listbox.yview)
        suggestions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.suggestions_listbox.config(yscrollcommand=suggestions_scrollbar.set)
      
        ttk.Button(
            search_frame, 
            text="Find Similar Foods", 
            command=self._find_similar_foods
        ).pack(fill=tk.X, pady=10)
       
        recommendations_frame = ttk.Frame(search_frame)
        recommendations_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(recommendations_frame, text="Number of recommendations:").pack(side=tk.LEFT)
        
        self.n_recommendations_var = tk.StringVar(value="5")
        n_recommendations_spinner = ttk.Spinbox(
            recommendations_frame, 
            from_=1, 
            to=20, 
            textvariable=self.n_recommendations_var, 
            width=5
        )
        n_recommendations_spinner.pack(side=tk.LEFT, padx=5)
        
        right_frame = ttk.Frame(self.similar_foods_tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.results_frame = ttk.LabelFrame(right_frame, text="Similar Foods", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
       
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, width=40, height=15)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
        
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
    
    def _setup_cluster_explorer_tab(self):
        """Set up the 'Cluster Explorer' tab."""
        
        top_frame = ttk.Frame(self.cluster_explorer_tab)
        top_frame.pack(fill=tk.X, pady=5)
       
        ttk.Label(top_frame, text="Select Cluster:").pack(side=tk.LEFT, padx=5)
        
        unique_clusters = sorted(list(set(self.labels)))
        self.selected_cluster_var = tk.StringVar()
        
        cluster_dropdown = ttk.Combobox(
            top_frame, 
            textvariable=self.selected_cluster_var, 
            values=[str(c) for c in unique_clusters],
            state="readonly",
            width=5
        )
        cluster_dropdown.pack(side=tk.LEFT, padx=5)
        cluster_dropdown.bind("<<ComboboxSelected>>", self._update_cluster_info)
       
        if unique_clusters:
            self.selected_cluster_var.set(str(unique_clusters[0]))
       
        bottom_frame = ttk.Frame(self.cluster_explorer_tab)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        left_pane = ttk.LabelFrame(bottom_frame, text="Cluster Information", padding=10)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.cluster_info_text = tk.Text(left_pane, wrap=tk.WORD, width=30, height=20)
        self.cluster_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cluster_info_text.config(state=tk.DISABLED)
        
        cluster_info_scrollbar = ttk.Scrollbar(left_pane, orient=tk.VERTICAL, command=self.cluster_info_text.yview)
        cluster_info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.cluster_info_text.config(yscrollcommand=cluster_info_scrollbar.set)
       
        right_pane = ttk.LabelFrame(bottom_frame, text="Foods in Cluster", padding=10)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.foods_listbox = tk.Listbox(right_pane, height=20, width=40)
        self.foods_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        foods_scrollbar = ttk.Scrollbar(right_pane, orient=tk.VERTICAL, command=self.foods_listbox.yview)
        foods_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.foods_listbox.config(yscrollcommand=foods_scrollbar.set)
       
        if unique_clusters:
            self._update_cluster_info(None)
    
    def _setup_visualizations_tab(self):
        """Set up the 'Visualizations' tab."""
        viz_container = ttk.Frame(self.visualizations_tab)
        viz_container.pack(fill=tk.BOTH, expand=True)
       
        left_pane = ttk.LabelFrame(viz_container, text="Available Visualizations", padding=10)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        viz_types = [
            ("Cluster Overview", "food_clusters.png", self._show_cluster_overview),
            ("Cluster Summary", "cluster_summary.png", self._show_cluster_summary),
            ("Interactive Plot", "clusters.html", self._show_interactive_plot)
        ]
        
        for viz_name, viz_file, viz_command in viz_types:
            viz_button = ttk.Button(
                left_pane, 
                text=viz_name, 
                command=viz_command,
                width=20
            )
            viz_button.pack(fill=tk.X, pady=5)
        
        self.viz_display = ttk.LabelFrame(viz_container, text="Visualization", padding=10)
        self.viz_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.viz_canvas_frame = ttk.Frame(self.viz_display)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.viz_canvas = tk.Canvas(self.viz_canvas_frame, bg="white")
        self.viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        viz_x_scrollbar = ttk.Scrollbar(self.viz_canvas_frame, orient=tk.HORIZONTAL, command=self.viz_canvas.xview)
        viz_x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        viz_y_scrollbar = ttk.Scrollbar(self.viz_canvas_frame, orient=tk.VERTICAL, command=self.viz_canvas.yview)
        viz_y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.viz_canvas.config(xscrollcommand=viz_x_scrollbar.set, yscrollcommand=viz_y_scrollbar.set)
        
        self.viz_canvas.create_text(
            400, 300,
            text="Select a visualization from the left panel",
            font=("Arial", 14)
        )
    
    def _update_suggestions(self, event):
        """Update the suggestions list based on user input."""
        search_text = self.search_var.get().lower()
        self.suggestions_listbox.delete(0, tk.END)
        
        if len(search_text) < 2:
            return
        
        matching_foods = self.food_names[self.food_names.str.lower().str.contains(search_text)].tolist()
        for food in matching_foods[:20]:  # Limit to 20 suggestions
            self.suggestions_listbox.insert(tk.END, food)
    
    def _select_suggestion(self, event):
        """Select a food from the suggestions list."""
        if not self.suggestions_listbox.curselection():
            return
        
        selection = self.suggestions_listbox.get(self.suggestions_listbox.curselection())
        self.search_var.set(selection)
        self.suggestions_listbox.delete(0, tk.END)
    
    def _find_similar_foods(self):
        """Find foods with similar nutrient profiles."""
        food_name = self.search_var.get()
        if not food_name:
            messagebox.showinfo("Input Required", "Please enter a food name")
            return
        
        try:
            n_recommendations = int(self.n_recommendations_var.get())
        except ValueError:
            n_recommendations = 5
        
        try:
            parent_dir = str(Path(__file__).resolve().parent)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
                
            from food_analysis import find_similar_foods
            
            result = find_similar_foods(food_name, n_recommendations)
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            if "error" in result:
                self.results_text.insert(tk.END, f"Error: {result['error']}\n")
                if "matches" in result:
                    self.results_text.insert(tk.END, "\nDid you mean one of these?\n")
                    for match in result["matches"]:
                        self.results_text.insert(tk.END, f"• {match}\n")
            else:
                self.results_text.insert(tk.END, f"Query Food: {result['query']}\n")
                self.results_text.insert(tk.END, f"Cluster: {result['cluster']}\n\n")
                self.results_text.insert(tk.END, "Similar Foods:\n")
                
                for i, food in enumerate(result['recommendations'], 1):
                    self.results_text.insert(tk.END, f"{i}. {food}\n")
            
            self.results_text.config(state=tk.DISABLED)
            
        except ImportError:
            self._find_similar_foods_internal(food_name, n_recommendations)
    
    def _find_similar_foods_internal(self, food_name, n_recommendations=5):
        """Internal implementation of find_similar_foods."""
        try:
            try:
                food_idx = self.food_names.str.lower().eq(food_name.lower()).idxmax()
                if not self.food_names.iloc[food_idx].lower() == food_name.lower():
                    matching_foods = self.food_names[self.food_names.str.lower().str.contains(food_name.lower())]
                    if len(matching_foods) == 0:
                        self.results_text.config(state=tk.NORMAL)
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, f"Error: Food '{food_name}' not found in database")
                        self.results_text.config(state=tk.DISABLED)
                        return
                    elif len(matching_foods) > 1:
                        self.results_text.config(state=tk.NORMAL)
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, f"Multiple matches found for '{food_name}'. Please select from the list below:\n\n")
                        for match in matching_foods:
                            self.results_text.insert(tk.END, f"• {match}\n")
                        self.results_text.config(state=tk.DISABLED)
                        return
                    food_idx = matching_foods.index[0]
            except:
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Error: Food '{food_name}' not found in database")
                self.results_text.config(state=tk.DISABLED)
                return
           
            food_cluster = self.labels[food_idx]
            cluster_indices = np.where(self.labels == food_cluster)[0]
            
            query_point = self.embedding[food_idx]
            distances = np.sqrt(np.sum((self.embedding[cluster_indices] - query_point)**2, axis=1))
            
            sorted_indices = np.argsort(distances)
            similar_food_indices = [cluster_indices[i] for i in sorted_indices if cluster_indices[i] != food_idx]
           
            recommendations = self.food_names.iloc[similar_food_indices[:n_recommendations]].tolist()
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            self.results_text.insert(tk.END, f"Query Food: {self.food_names.iloc[food_idx]}\n")
            self.results_text.insert(tk.END, f"Cluster: {food_cluster}\n\n")
            self.results_text.insert(tk.END, "Similar Foods:\n")
            
            for i, food in enumerate(recommendations, 1):
                self.results_text.insert(tk.END, f"{i}. {food}\n")
            
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error finding similar foods: {e}")
            traceback.print_exc()
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}")
            self.results_text.config(state=tk.DISABLED)
    
    def _update_cluster_info(self, event):
        """Update the cluster information when a cluster is selected."""
        try:
            selected_cluster = int(self.selected_cluster_var.get())
        except ValueError:
            return
       
        self.cluster_info_text.config(state=tk.NORMAL)
        self.cluster_info_text.delete(1.0, tk.END)
        
        cluster_size = sum(self.labels == selected_cluster)
        self.cluster_info_text.insert(tk.END, f"Cluster: {selected_cluster}\n")
        self.cluster_info_text.insert(tk.END, f"Number of foods: {cluster_size}\n\n")
        
        try:
            if hasattr(self, 'cluster_data'):
                cluster_data = self.cluster_data[self.cluster_data['cluster'] == selected_cluster]
                
                center_x = cluster_data['x'].mean()
                center_y = cluster_data['y'].mean()
                
                self.cluster_info_text.insert(tk.END, f"Cluster Center: ({center_x:.2f}, {center_y:.2f})\n\n")
                self.cluster_info_text.insert(tk.END, "Sample Foods:\n")
               
                for _, row in cluster_data.head(5).iterrows():
                    self.cluster_info_text.insert(tk.END, f"• {row['food_name']}\n")
                    
            elif hasattr(self, 'cluster_summary'):
                if selected_cluster in self.cluster_summary.index:
                    cluster_summary = self.cluster_summary.loc[selected_cluster]
                    self.cluster_info_text.insert(tk.END, "Nutrient Profile (avg. values):\n")
                  
                    for nutrient, value in cluster_summary.items():
                        if nutrient not in ['examples', 'size']:
                            self.cluster_info_text.insert(tk.END, f"{nutrient}: {value:.4f}\n")
        except Exception as e:
            logger.warning(f"Could not display detailed cluster info: {e}")
        
        self.cluster_info_text.config(state=tk.DISABLED)
       
        self.foods_listbox.delete(0, tk.END)
        
        cluster_foods = self.food_names[self.labels == selected_cluster].tolist()
        for food in cluster_foods:
            self.foods_listbox.insert(tk.END, food)
    
    def _show_cluster_overview(self):
        """Display the cluster overview visualization."""
        img_path = "docs/images/food_clusters.png"
        self._display_image(img_path)
    
    def _show_cluster_summary(self):
        """Display the cluster summary visualization."""
        img_path = "docs/images/cluster_summary.png"
        self._display_image(img_path)
    
    def _show_interactive_plot(self):
        """Open the interactive plot in the default web browser."""
        html_path = "docs/html/clusters.html"
        if os.path.exists(html_path):
            webbrowser.open('file://' + os.path.abspath(html_path))
        else:
            messagebox.showinfo("File Not Found", 
                               "Interactive visualization not found. Please run the analysis first.")
    
    def _display_image(self, img_path):
        """Display an image on the canvas."""
        if not os.path.exists(img_path):
            messagebox.showinfo("File Not Found", 
                               f"Image '{img_path}' not found. Please run the analysis first.")
            return
        
        self.viz_canvas.delete("all")
       
        try:
            img = Image.open(img_path)
            self.displayed_image = ImageTk.PhotoImage(img)
            
            self.viz_canvas.create_image(0, 0, anchor=tk.NW, image=self.displayed_image)
           
            self.viz_canvas.config(scrollregion=self.viz_canvas.bbox(tk.ALL))
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

def main():
    """Run the Food Explorer application."""
    try:
        root = tk.Tk()
        app = FoodExplorerApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()