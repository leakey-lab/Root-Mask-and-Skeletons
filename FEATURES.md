# Root Analysis Application: Features for Plant Biologists

This document outlines the key features of the Root-Mask-and-Skeletons (SPROUTS) application, specifically tailored to the needs of plant biologists conducting root analysis.

### I. Core Analysis Workflow

The application provides a powerful, end-to-end workflow for processing root images, from initial analysis to data export. You can choose to generate segmentation masks for area analysis or create detailed skeletons for morphometric measurements.

```mermaid
graph TD
    subgraph "1. Input"
        A["📷 Raw Image"]
    end

    subgraph "2. Analysis Path"
        B{Processing Choice};
        A --> B;
        
        subgraph "Segmentation"
            B -- "Mask Generation" --> C["🎭 Generate Mask<br/>(Automated with ResNet ML)"];
            C --> D["✏️ Manually Edit & Refine Mask"];
            D --> E["💾 Save Final Mask"];
        end

        subgraph "Skeletonization"
            B -- "Skeleton Generation" --> F["🦴 Generate Skeleton<br/>(Automated with Pix2Pix GAN)"];
            F --> G["📏 Calculate Metrics<br/>(Length, Area, etc.)"];
            G --> H["📊 Export Data to CSV"];
        end
    end
    
    subgraph "3. Outputs"
        E --> I[("🎭 Binary Mask File")]
        F --> J[("💀 Skeleton Image File")]
        H --> K[("📈 CSV Metrics File")]
    end
```

**Key Features:**
-   **Dual ML Models**: Employs a ResNet model for accurate segmentation and a Pix2Pix GAN for high-fidelity skeletonization.
-   **Interactive Editing**: Provides a full suite of tools (brush, eraser, fill) for fine-tuning ML-generated masks.
-   **Rich Data Export**: Exports calculated metrics to CSV for easy integration with external statistical analysis tools.

### II. Interactive Data Visualization

Once your root metrics are calculated, the application offers an integrated, interactive dashboard to explore and understand your results without leaving the application.

```mermaid
graph TD
    A["📊 Metrics Data (CSV)"] --> B["📈 Load Data into Dashboard"];
    B --> C["🎨 Generate Interactive Visualizations"];
    
    subgraph "Available Charts"
        C --> D["Scatter Plots"];
        C --> E["Box Plots"];
        C --> F["Histograms"];
    end

    subgraph "Interactivity"
        C --> G["🔬 Filter by Metadata<br/>(e.g., Treatment, Date)"];
        C --> H["📂 Group Data"];
    end
```

### III. Efficient Workflow and Data Management

Designed for high-throughput experiments, the application includes features to manage large datasets efficiently.

```mermaid
graph TD
    A["📁 Select Folder of Images"] --> B["⚙️ High-Throughput Batch Processing"];
    B --> C["🌳 Automatically Organizes Files<br/>Based on Parsed Filename Metadata"];
    subgraph "Organized Hierarchy"
        C --> D["- Field Site"]
        D --> E["- Date"]
        E --> F["- Tube ID"]
    end
    C --> G["⚡ Quick Navigation & Access"];
    B --> H["📤 Structured & Predictable Output Folders"];
```

**Key Features:**
-   **Hierarchical Image Organization**: Parses filenames to create a navigable tree view, organizing images by metadata like field, date, and tube number.
-   **Hardware Optimization**: Intelligently utilizes available GPU (CUDA) for ML tasks to accelerate processing, with a seamless fallback to CPU.
-   **Intuitive UI**: A modern and user-friendly interface built with PyQt6.


### IV. Extensible and Performant Architecture

The application is built on a modular and performance-oriented architecture, ensuring it is both powerful and future-proof.

```mermaid
graph TD
    subgraph "Core System"
        A["🧱 Modular & Decoupled<br/>(Frontend/Backend)"]
    end

    subgraph "Benefits"
        A -- "Allows for" --> B["🧩 Easy Extensibility<br/>(New Models & Modules)"];
        A -- "Ensures" --> C["🚀 High Performance<br/>(Multithreading & Subprocessing)"];
        C --> D["💻 Responsive User Interface"];
    end
```
