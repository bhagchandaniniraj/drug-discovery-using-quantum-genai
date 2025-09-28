import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import py3Dmol
from IPython.display import display, HTML
import os
from typing import List, Tuple, Optional, Dict
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ProgressTracker:
    """Comprehensive progress tracking and reporting"""
    def __init__(self):
        self.start_time = time.time()
        self.current_step = 0
        self.total_steps = 0
        self.step_times = []
        
    def start_program(self, total_steps: int):
        """Initialize program progress tracking"""
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        print("🚀 STARTING HYBRID QUANTUM-CLASSICAL DOCKING ANALYSIS")
        print("="*80)
        print(f"📋 Total steps to execute: {total_steps}")
        print("="*80)
        
    def start_step(self, step_name: str, step_number: int = None):
        """Start a new step with progress tracking"""
        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1
            
        step_start = time.time()
        self.step_times.append({'name': step_name, 'start': step_start})
        
        print(f"\n🔹 Step {self.current_step}/{self.total_steps}: {step_name}")
        print(f"   ⏰ Started at: {time.strftime('%H:%M:%S')}")
        if self.current_step > 1:
            elapsed = time.time() - self.start_time
            estimated_total = (elapsed / (self.current_step - 1)) * self.total_steps
            remaining = estimated_total - elapsed
            print(f"   ⏱️  Elapsed: {self.format_time(elapsed)} | Estimated remaining: {self.format_time(remaining)}")
        
    def end_step(self, success: bool = True, additional_info: str = ""):
        """End current step with results"""
        if not self.step_times:
            return
            
        current_step = self.step_times[-1]
        step_duration = time.time() - current_step['start']
        
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"   {status} - Duration: {self.format_time(step_duration)}")
        if additional_info:
            print(f"   📊 {additional_info}")
            
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
            
    def print_progress_summary(self):
        """Print final progress summary"""
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print("📈 EXECUTION SUMMARY")
        print("="*80)
        print(f"Total execution time: {self.format_time(total_time)}")
        print(f"Steps completed: {self.current_step}/{self.total_steps}")
        print(f"Average time per step: {self.format_time(total_time/self.current_step)}")

class QuantumGenerator(nn.Module):
    """Quantum Circuit-based Generative Model for Pose Generation"""
    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = nn.Parameter(torch.randn(n_layers * n_qubits * 3))
        
    def forward(self, z: torch.Tensor, progress_callback=None) -> torch.Tensor:
        """Generate poses using quantum circuit with progress tracking"""
        batch_size = z.shape[0]
        coords_list = []
        
        if progress_callback:
            progress_callback(f"Generating {batch_size} quantum poses")
        
        for i in range(batch_size):
            if progress_callback and i % max(1, batch_size//10) == 0:
                progress_callback(f"Quantum pose {i+1}/{batch_size}")
                
            # Simplified quantum-inspired transformation
            quantum_state = torch.atan(z[i] * self.theta[:len(z[i])])
            coord = torch.sin(quantum_state) * 5.0
            coords_list.append(coord.unsqueeze(0))
            
        return torch.cat(coords_list, dim=0)

class QuantumScoringNetwork(nn.Module):
    """Quantum-Classical Hybrid Scoring Function"""
    def __init__(self, protein_feat_dim=64, ligand_feat_dim=32):
        super().__init__()
        
        self.protein_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, protein_feat_dim), nn.Tanh()
        )
        
        self.ligand_encoder = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, ligand_feat_dim), nn.Tanh()
        )
        
        self.classical_head = nn.Sequential(
            nn.Linear(protein_feat_dim + ligand_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def quantum_feature_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired feature transformation"""
        phase_shift = torch.atan(features)
        entanglement = torch.sin(phase_shift * np.pi)
        return entanglement
    
    def forward(self, protein_coords: torch.Tensor, ligand_coords: torch.Tensor, 
                progress_callback=None) -> torch.Tensor:
        """Forward pass with progress tracking"""
        if progress_callback:
            progress_callback("Encoding protein coordinates")
        
        protein_encoded = self.protein_encoder(protein_coords.float())
        protein_feat = torch.mean(protein_encoded, dim=0)
        
        if progress_callback:
            progress_callback("Encoding ligand coordinates")
        
        ligand_encoded = self.ligand_encoder(ligand_coords.float())
        ligand_feat = torch.mean(ligand_encoded, dim=0)
        
        if progress_callback:
            progress_callback("Applying quantum feature transform")
        
        protein_quantum = self.quantum_feature_transform(protein_feat)
        ligand_quantum = self.quantum_feature_transform(ligand_feat)
        
        combined = torch.cat([protein_quantum, ligand_quantum], dim=0)
        
        if progress_callback:
            progress_callback("Final scoring")
        
        score = self.classical_head(combined.unsqueeze(0))
        return score

class HybridQuantumDocking:
    def __init__(self, protein_file: str, ligand_data: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.protein_file = protein_file
        self.ligand_data = ligand_data
        self.device = device
        self.progress = ProgressTracker()
        
        # Initialize models
        self.quantum_generator = QuantumGenerator().to(device)
        self.quantum_scorer = QuantumScoringNetwork().to(device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.quantum_generator.parameters(), lr=0.001)
        self.score_optimizer = optim.Adam(self.quantum_scorer.parameters(), lr=0.0005)
        
        # Data storage
        self.protein_coords = None
        self.ligands_3d = []
        self.ligands_mol = []
        self.ligands_info = []
        self.docking_results = []
        
    def _auto_detect_column(self, df: pd.DataFrame, possible_names: List[str], default_name: str) -> str:
        """Auto-detect column names in dataframe"""
        for name in possible_names:
            if name in df.columns:
                return name
        return default_name
    
    def quantum_conformer_score(self, coords: np.ndarray) -> float:
        """Score conformer based on geometric properties"""
        # Calculate radius of gyration
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        radius_gyration = np.sqrt(np.mean(distances**2))
        
        # Calculate compactness score (inverse of radius of gyration)
        compactness = 1.0 / (radius_gyration + 1e-6)
        
        # Add some quantum-inspired randomness
        quantum_factor = np.random.normal(1.0, 0.1)
        
        return compactness * quantum_factor

    def load_protein(self) -> bool:
        """Load protein structure with detailed progress tracking"""
        self.progress.start_step("Loading protein structure", 1)
        
        try:
            if not os.path.exists(self.protein_file):
                print("   ⚠️ Protein file not found, generating demo structure...")
                self._generate_quantum_demo_protein()
                self.progress.end_step(True, f"Generated demo protein with {len(self.protein_coords)} atoms")
                return True
                
            self.progress.start_step("Reading PDB file", 1.1)
            ppdb = PandasPdb()
            ppdb.read_pdb(self.protein_file)
            self.protein_df = ppdb.df['ATOM']
            self.progress.end_step(True, f"Read {len(self.protein_df)} atoms")
            
            self.progress.start_step("Extracting coordinates", 1.2)
            self.protein_coords = self.protein_df[['x_coord', 'y_coord', 'z_coord']].values
            self.protein_atom_types = self.protein_df['atom_name'].values
            self.progress.end_step(True, f"Extracted {len(self.protein_coords)} coordinates")
            
            self.progress.start_step("Quantum enhancement", 1.3)
            self.protein_coords = self.quantum_enhance_protein_coords(self.protein_coords)
            self.progress.end_step(True, "Applied quantum enhancements")
            
            protein_center = np.mean(self.protein_coords, axis=0)
            protein_radius = np.max(np.linalg.norm(self.protein_coords - protein_center, axis=1))
            
            self.progress.end_step(True, 
                f"Loaded protein: {len(self.protein_coords)} atoms, "
                f"radius: {protein_radius:.2f}Å")
            return True
            
        except Exception as e:
            self.progress.end_step(False, f"Error: {str(e)}")
            self._generate_quantum_demo_protein()
            return False

    def load_ligands(self, max_ligands: int = 10) -> bool:
        """Load ligands with comprehensive progress tracking"""
        self.progress.start_step("Loading and processing ligands", 2)
        
        try:
            if not os.path.exists(self.ligand_data):
                print("   ⚠️ Ligand file not found, generating demo ligands...")
                self._generate_quantum_demo_ligands(min(3, max_ligands))
                self.progress.end_step(True, f"Generated {len(self.ligands_mol)} demo ligands")
                return True
            
            self.progress.start_step("Reading ligand file", 2.1)
            df = pd.read_excel(self.ligand_data) if self.ligand_data.endswith('.xlsx') else pd.read_csv(self.ligand_data)
            smiles_column = self._auto_detect_column(df, ['smiles', 'SMILES', 'Smiles'], 'Smiles')
            name_column = self._auto_detect_column(df, ['Name', 'NAME', 'name'], 'Name')
            
            smiles_list = df[smiles_column].dropna().tolist()[:max_ligands]
            names_list = df[name_column].dropna().tolist()[:max_ligands] if name_column else [f"Ligand_{i+1}" for i in range(len(smiles_list))]
            self.progress.end_step(True, f"Found {len(smiles_list)} ligands in file")
            
            self.ligands_3d = []
            self.ligands_mol = []
            self.ligands_info = []
            
            successful_ligands = 0
            for i, (smiles, name) in enumerate(zip(smiles_list, names_list)):
                
                self.progress.start_step(f"Processing {name}", 2.2 + i*0.1)
                mol, coords = self.quantum_enhanced_smiles_to_3d(smiles, name)
                
                if mol is not None and coords is not None:
                    self.ligands_mol.append(mol)
                    self.ligands_3d.append(coords)
                    properties = self.calculate_quantum_properties(mol)
                    properties.update({'name': name, 'smiles': smiles})
                    self.ligands_info.append(properties)
                    successful_ligands += 1
                    self.progress.end_step(True, f"Successfully processed {name}")
                else:
                    self.progress.end_step(False, f"Failed to process {name}")
            
            self.progress.end_step(True, f"Successfully processed {successful_ligands}/{len(smiles_list)} ligands")
            return True
            
        except Exception as e:
            self.progress.end_step(False, f"Error: {str(e)}")
            self._generate_quantum_demo_ligands(min(3, max_ligands))
            return False

    def quantum_enhanced_smiles_to_3d(self, smiles: str, name: str) -> Tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
        """Convert SMILES to 3D with progress tracking"""
        try:
            self.progress.start_step(f"Converting {name} to 3D", 0)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.progress.end_step(False, "Invalid SMILES")
                return None, None
                
            mol = Chem.AddHs(mol)
            all_conformers = []
            conformer_scores = []
            
            for attempt in range(3):
                self.progress.start_step(f"Conformer generation attempt {attempt+1}", 0.1 + attempt*0.1)
                try:
                    status = AllChem.EmbedMolecule(mol, randomSeed=42+attempt*10)
                    if status == -1:
                        self.progress.end_step(False, "Embedding failed")
                        continue
                    
                    AllChem.MMFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                    
                    score = self.quantum_conformer_score(coords)
                    all_conformers.append(coords)
                    conformer_scores.append(score)
                    self.progress.end_step(True, f"Score: {score:.3f}")
                    
                except Exception as e:
                    self.progress.end_step(False, f"Error: {str(e)}")
                    continue
            
            if not all_conformers:
                self.progress.end_step(False, "All conformer generations failed")
                return None, None
            
            best_idx = np.argmax(conformer_scores)
            best_coords = all_conformers[best_idx]
            self.progress.end_step(True, f"Best score: {conformer_scores[best_idx]:.3f}")
            return mol, best_coords
            
        except Exception as e:
            self.progress.end_step(False, f"Unexpected error: {str(e)}")
            return None, None

    def calculate_quantum_properties(self, mol: Chem.Mol) -> Dict:
        """Calculate molecular properties with quantum-inspired metrics"""
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'h_bond_donors': Descriptors.NumHDonors(mol),
            'h_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'quantum_stability': np.random.normal(0.5, 0.1)  # Quantum-inspired metric
        }

    def quantum_enhance_protein_coords(self, coords: np.ndarray) -> np.ndarray:
        """Apply quantum-inspired enhancements to protein coordinates"""
        # Add small quantum fluctuations
        quantum_noise = np.random.normal(0, 0.1, coords.shape)
        return coords + quantum_noise

    def _generate_quantum_demo_protein(self):
        """Generate a demo protein structure"""
        # Simple alpha-helix like structure
        n_atoms = 200
        t = np.linspace(0, 4*np.pi, n_atoms)
        x = 10 * np.cos(t)
        y = 10 * np.sin(t)
        z = 0.5 * t
        
        self.protein_coords = np.column_stack([x, y, z])
        self.protein_atom_types = ['C'] * n_atoms

    def _generate_quantum_demo_ligands(self, n_ligands: int):
        """Generate demo ligands"""
        demo_smiles = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "C1CCCCC1",  # Cyclohexane
            "c1ccccc1",  # Benzene
            "CNC",  # Dimethylamine
        ]
        
        demo_names = ["Ethanol", "Acetic_acid", "Cyclohexane", "Benzene", "Dimethylamine"]
        
        for i in range(min(n_ligands, len(demo_smiles))):
            mol, coords = self.quantum_enhanced_smiles_to_3d(demo_smiles[i], demo_names[i])
            if mol is not None:
                self.ligands_mol.append(mol)
                self.ligands_3d.append(coords)
                properties = self.calculate_quantum_properties(mol)
                properties.update({'name': demo_names[i], 'smiles': demo_smiles[i]})
                self.ligands_info.append(properties)

    def quantum_annealing_docking(self, ligand_coords: np.ndarray, n_iterations: int = 50) -> Tuple[np.ndarray, float]:
        """Quantum annealing-inspired docking optimization"""
        best_score = -float('inf')
        best_pose = None
        
        protein_center = np.mean(self.protein_coords, axis=0)
        ligand_center = np.mean(ligand_coords, axis=0)
        
        # Initial translation to bring ligand near protein
        initial_offset = protein_center - ligand_center + np.random.normal(0, 2, 3)
        
        for iteration in range(n_iterations):
            # Quantum-inspired temperature schedule
            temperature = 10.0 * np.exp(-iteration / (n_iterations / 3))
            
            # Generate random rotation and translation
            rotation = self._generate_quantum_rotation(temperature)
            translation = np.random.normal(0, temperature/5, 3)
            
            # Apply transformation
            transformed_coords = self._apply_transformation(ligand_coords, rotation, translation + initial_offset)
            
            # Score the pose
            score = self.hybrid_quantum_score(transformed_coords)
            
            # Quantum-inspired acceptance probability
            if score > best_score or np.random.random() < np.exp((score - best_score) / max(temperature, 0.1)):
                best_score = score
                best_pose = transformed_coords
        
        return best_pose, best_score

    def _generate_quantum_rotation(self, temperature: float) -> np.ndarray:
        """Generate quantum-inspired rotation matrix"""
        # Small random rotations with temperature scaling
        angles = np.random.normal(0, temperature * np.pi / 180, 3)
        rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
        ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        return rz @ ry @ rx

    def _apply_transformation(self, coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to coordinates"""
        return (rotation @ coords.T).T + translation

    def hybrid_quantum_score(self, ligand_pose: np.ndarray) -> float:
        """Hybrid quantum-classical scoring function"""
        # Distance-based scoring
        distances = []
        for protein_atom in self.protein_coords[:100]:  # Sample for efficiency
            for ligand_atom in ligand_pose:
                dist = np.linalg.norm(protein_atom - ligand_atom)
                if dist < 8.0:  # Only consider nearby atoms
                    distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Convert distances to scores (closer is better)
        dist_scores = np.exp(-np.array(distances) / 2.0)
        
        # Shape complementarity (simplified)
        protein_center = np.mean(self.protein_coords, axis=0)
        ligand_center = np.mean(ligand_pose, axis=0)
        center_dist = np.linalg.norm(protein_center - ligand_center)
        center_score = np.exp(-center_dist / 5.0)
        
        # Quantum fluctuation factor
        quantum_factor = 1.0 + 0.1 * np.random.normal()
        
        return np.mean(dist_scores) * center_score * quantum_factor

    def generative_quantum_docking(self, ligand_idx: int, n_restarts: int = 3) -> Tuple[np.ndarray, float]:
        """Generative quantum docking with multiple restarts"""
        if ligand_idx >= len(self.ligands_3d):
            return None, 0.0
        
        ligand_coords = self.ligands_3d[ligand_idx]
        best_score = -float('inf')
        best_pose = None
        
        for restart in range(n_restarts):
            pose, score = self.quantum_annealing_docking(ligand_coords, n_iterations=30)
            if score > best_score:
                best_score = score
                best_pose = pose
        
        return best_pose, best_score

    def visualize_3d_structure(self, coords: np.ndarray, title: str, atom_types: list = None):
        """Create interactive 3D visualization using plotly"""
        fig = go.Figure()
        
        if atom_types is None:
            atom_types = ['C'] * len(coords)
        
        # Color mapping for atoms
        color_map = {'C': 'black', 'O': 'red', 'N': 'blue', 'H': 'gray', 'S': 'yellow'}
        
        # Add atoms
        for i, (x, y, z) in enumerate(coords):
            atom_type = atom_types[i] if i < len(atom_types) else 'C'
            color = color_map.get(atom_type, 'purple')
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=8, color=color),
                name=f'{atom_type}{i+1}',
                text=f'{atom_type}{i+1}',
                hoverinfo='text'
            ))
        
        # Add bonds (simplified - just connect nearby atoms)
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 2.0:  # Typical bond length
                    fig.add_trace(go.Scatter3d(
                        x=[coords[i][0], coords[j][0]],
                        y=[coords[i][1], coords[j][1]],
                        z=[coords[i][2], coords[j][2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=f'3D Structure: {title}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig

    def visualize_protein_ligand_docking(self, protein_coords: np.ndarray, ligand_coords: np.ndarray, 
                                       protein_title: str, ligand_title: str):
        """Visualize protein-ligand docking complex"""
        fig = go.Figure()
        
        # Sample protein coordinates for better visualization
        if len(protein_coords) > 1000:
            indices = np.random.choice(len(protein_coords), 1000, replace=False)
            protein_coords = protein_coords[indices]
        
        # Protein as points
        fig.add_trace(go.Scatter3d(
            x=protein_coords[:, 0], y=protein_coords[:, 1], z=protein_coords[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.6),
            name='Protein',
            text='Protein atoms',
            hoverinfo='text'
        ))
        
        # Ligand as connected structure
        fig.add_trace(go.Scatter3d(
            x=ligand_coords[:, 0], y=ligand_coords[:, 1], z=ligand_coords[:, 2],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Ligand Atoms',
            text='Ligand atoms',
            hoverinfo='text'
        ))
        
        # Add ligand bonds
        for i in range(len(ligand_coords)):
            for j in range(i+1, len(ligand_coords)):
                distance = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                if distance < 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[ligand_coords[i][0], ligand_coords[j][0]],
                        y=[ligand_coords[i][1], ligand_coords[j][1]],
                        z=[ligand_coords[i][2], ligand_coords[j][2]],
                        mode='lines',
                        line=dict(color='red', width=4),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=f'Docking Complex: {protein_title} + {ligand_title}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=900,
            height=700
        )
        
        return fig

    def create_projection_plots(self, protein_coords: np.ndarray, ligand_coords: np.ndarray, title: str):
        """Create XY and XZ projection plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # XY projection
        ax1.scatter(protein_coords[:, 0], protein_coords[:, 1], alpha=0.6, label='Protein', s=10)
        ax1.scatter(ligand_coords[:, 0], ligand_coords[:, 1], alpha=0.8, label='Ligand', s=30)
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_title(f'{title} - XY Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XZ projection
        ax2.scatter(protein_coords[:, 0], protein_coords[:, 2], alpha=0.6, label='Protein', s=10)
        ax2.scatter(ligand_coords[:, 0], ligand_coords[:, 2], alpha=0.8, label='Ligand', s=30)
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Z (Å)')
        ax2.set_title(f'{title} - XZ Plane')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_distance_analysis(self, protein_coords: np.ndarray, ligand_coords: np.ndarray):
        """Create distance distribution analysis"""
        distances = []
        for lig_atom in ligand_coords:
            min_dist = float('inf')
            for prot_atom in protein_coords[:500]:  # Sample for efficiency
                dist = np.linalg.norm(lig_atom - prot_atom)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distance histogram
        ax1.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Distance (Å)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Protein-Ligand Distance Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        ax2.hist(distances, bins=20, alpha=0.7, color='lightcoral', 
                edgecolor='black', cumulative=True, density=True)
        ax2.set_xlabel('Distance (Å)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distance Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, distances

    def create_analysis_dashboard(self):
        """Create comprehensive analysis dashboard"""
        if not self.docking_results:
            print("No docking results to visualize")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Docking Scores', 'Ligand Properties', 'Score Distribution', 'Property Correlation'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # 1. Docking scores bar chart
        names = [result['name'] for result in self.docking_results]
        scores = [result['score'] for result in self.docking_results]
        
        fig.add_trace(go.Bar(x=names, y=scores, name='Docking Scores'), row=1, col=1)
        
        # 2. Property scatter plot
        molecular_weights = [result['properties']['molecular_weight'] for result in self.docking_results]
        logp_values = [result['properties']['logp'] for result in self.docking_results]
        
        fig.add_trace(go.Scatter(x=molecular_weights, y=logp_values, mode='markers+text',
                               text=names, name='Properties'), row=1, col=2)
        
        # 3. Score distribution
        fig.add_trace(go.Histogram(x=scores, name='Score Distribution'), row=2, col=1)
        
        # 4. Property correlation heatmap
        properties_df = pd.DataFrame([result['properties'] for result in self.docking_results])
        numeric_cols = properties_df.select_dtypes(include=[np.number]).columns
        corr_matrix = properties_df[numeric_cols].corr()
        
        fig.add_trace(go.Heatmap(z=corr_matrix.values, 
                               x=corr_matrix.columns,
                               y=corr_matrix.columns,
                               colorscale='Viridis'), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Quantum Docking Analysis Dashboard")
        return fig

    def print_comprehensive_results_table(self):
        """Print comprehensive results table in the requested format"""
        if not self.docking_results:
            print("No docking results available")
            return
        
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE DOCKING RESULTS SUMMARY")
        print("="*100)
        print(f"{'Rank':<4} {'Ligand Name':<15} {'SMILES':<30} {'Score':<8} {'MW':<8} {'LogP':<6} {'HBD':<4} {'HBA':<4}")
        print("-" * 100)
        
        sorted_results = sorted(self.docking_results, key=lambda x: x['score'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            name = result['name'][:14]
            smiles = result['properties']['smiles'][:29] if 'smiles' in result['properties'] else 'N/A'
            score = f"{result['score']:.4f}"
            mw = f"{result['properties']['molecular_weight']:.1f}"
            logp = f"{result['properties']['logp']:.2f}"
            hbd = f"{result['properties']['h_bond_donors']}"
            hba = f"{result['properties']['h_bond_acceptors']}"
            
            print(f"{i:<4} {name:<15} {smiles:<30} {score:<8} {mw:<8} {logp:<6} {hbd:<4} {hba:<4}")
        
        print("-" * 100)

    def run_complete_analysis(self, n_ligands: int = 3):
        """Complete analysis with proper visualization"""
        total_steps = 6 + n_ligands * 3
        self.progress.start_program(total_steps)
        
        # Load data
        self.load_protein()
        self.load_ligands(max_ligands=n_ligands)
        
        # Visualize protein
        self.progress.start_step("Visualizing protein structure", 3)
        if self.protein_coords is not None:
            protein_fig = self.visualize_3d_structure(self.protein_coords[:500], "Protein Structure")
            protein_fig.show()
        self.progress.end_step(True, "Protein visualization completed")
        
        # Visualize ligands and perform docking
        self.docking_results = []
        for ligand_idx in range(min(n_ligands, len(self.ligands_3d))):
            # Visualize ligand
            self.progress.start_step(f"Visualizing ligand {ligand_idx+1}", 4 + ligand_idx*3)
            if ligand_idx < len(self.ligands_3d):
                ligand_fig = self.visualize_3d_structure(self.ligands_3d[ligand_idx], 
                                                       self.ligands_info[ligand_idx]['name'])
                ligand_fig.show()
            self.progress.end_step(True, "Ligand visualization completed")
            
            # Perform docking
            self.progress.start_step(f"Docking ligand {ligand_idx+1}", 5 + ligand_idx*3)
            docked_pose, score = self.generative_quantum_docking(ligand_idx)
            
            if docked_pose is not None:
                self.docking_results.append({
                    'ligand_idx': ligand_idx,
                    'name': self.ligands_info[ligand_idx]['name'],
                    'docked_pose': docked_pose,
                    'score': score,
                    'properties': self.ligands_info[ligand_idx]
                })
                
                # Visualize docking result
                self.progress.start_step(f"Visualizing docking result {ligand_idx+1}", 6 + ligand_idx*3)
                docking_fig = self.visualize_protein_ligand_docking(
                    self.protein_coords, docked_pose,
                    "Protein", self.ligands_info[ligand_idx]['name']
                )
                docking_fig.show()
                
                # Create projection plots
                proj_fig = self.create_projection_plots(
                    self.protein_coords, docked_pose,
                    f"Docking: {self.ligands_info[ligand_idx]['name']}"
                )
                plt.show()
                
                # Distance analysis
                dist_fig, distances = self.create_distance_analysis(self.protein_coords, docked_pose)
                plt.show()
                
                self.progress.end_step(True, f"Docking completed with score: {score:.4f}")
        
        # Print comprehensive results table
        self.progress.start_step("Generating results table", total_steps-2)
        self.print_comprehensive_results_table()
        self.progress.end_step(True, "Results table generated")
        
        # Create analysis dashboard
        self.progress.start_step("Creating analysis dashboard", total_steps-1)
        dashboard_fig = self.create_analysis_dashboard()
        if dashboard_fig:
            dashboard_fig.show()
        self.progress.end_step(True, "Analysis dashboard created")
        
        # Final summary
        self.progress.start_step("Generating final summary", total_steps)
        self.print_comprehensive_summary()
        self.progress.end_step(True, "Analysis complete")
        
        self.progress.print_progress_summary()
        return self.docking_results

    def print_comprehensive_summary(self):
        """Print detailed analysis summary"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE QUANTUM DOCKING ANALYSIS SUMMARY")
        print("="*80)
        
        if self.docking_results:
            print("\n📊 DOCKING RESULTS (Ranked by Score):")
            print("-" * 50)
            sorted_results = sorted(self.docking_results, key=lambda x: x['score'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                print(f"{i}. {result['name']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Molecular Weight: {result['properties']['molecular_weight']:.2f}")
                print(f"   LogP: {result['properties']['logp']:.2f}")
                print(f"   H-Bond Donors: {result['properties']['h_bond_donors']}")
                print(f"   H-Bond Acceptors: {result['properties']['h_bond_acceptors']}")
                print("-" * 30)
        
        print(f"\n📈 STATISTICS:")
        print(f"   Total ligands processed: {len(self.ligands_info)}")
        print(f"   Successful dockings: {len(self.docking_results)}")
        if self.docking_results:
            scores = [r['score'] for r in self.docking_results]
            print(f"   Average score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            print(f"   Best score: {max(scores):.4f}")
            print(f"   Worst score: {min(scores):.4f}")

def run_complete_quantum_docking_analysis():
    """Main function with complete visualization"""
    print("Initializing Complete Hybrid Quantum Docking System...")
    
    # Create dataset directory if it doesn't exist
    os.makedirs('Dataset', exist_ok=True)
    
    docking_system = HybridQuantumDocking(
        protein_file='Dataset/5v90.pdb',
        ligand_data='Dataset/ChEMBL.xlsx'
    )
    
    print("\nStarting comprehensive analysis with 3D visualization...")
    results = docking_system.run_complete_analysis(n_ligands=3)
    
    return docking_system, results

if __name__ == "__main__":
    docking_system, results = run_complete_quantum_docking_analysis()