import subprocess

def generar_diagrama_mermaid(codigo_diagrama, nombre_archivo):
    with open(f"{nombre_archivo}.mmd", "w") as f:
        f.write(codigo_diagrama)

    # Genera el archivo PNG utilizando la CLI
    subprocess.run([
        "mmdc", "-i", f"{nombre_archivo}.mmd", "-o", f"{nombre_archivo}.svg",
        "--scale", "5",  # Aumentar la escala
        "--width", "1600",  # Ajustar el ancho
        "--height", "1000"  # Ajustar el alto
    ])

mermaid_code = """
graph TD
    A[Bot Creator] --> B(.env)
    A --> C(config)
    A --> D(docker-compose.yml)
    D --> E[Docker Bot 1]
    D --> F[Docker Bot 2]
    D --> G[Docker Bot 3]
    E --> H[API Dash]
    F --> H
    G --> H
    H --> I[Frontend]

"""
generar_diagrama_mermaid(mermaid_code, "LTD_Flow")