# HDFS Manager
# Responsável: Ana Luiza Pazze
# Gerenciamento do HDFS para armazenamento distribuído

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class HDFSManager:
    """
    Gerenciador do HDFS para armazenamento distribuído de dados financeiros
    """
    
    def __init__(self, 
                 hdfs_host: str = None,
                 hdfs_port: int = None,
                 hdfs_user: str = "hadoop"):
        """
        Inicializa o gerenciador HDFS
        
        Args:
            hdfs_host: Host do HDFS
            hdfs_port: Porta do HDFS
            hdfs_user: Usuário HDFS
        """
        # Permitir configuração via variável de ambiente HDFS_NAMENODE_URL (ex.: hdfs://namenode:9000)
        env_url = os.getenv("HDFS_NAMENODE_URL")
        if env_url:
            # Extrair host e porta se possível
            try:
                # hdfs://host:port
                without_scheme = env_url.replace("hdfs://", "")
                host_part, port_part = without_scheme.split(":")
                self.hdfs_host = host_part
                self.hdfs_port = int(port_part)
            except Exception:
                # Fallback
                self.hdfs_host = hdfs_host or "localhost"
                self.hdfs_port = hdfs_port or 9000
        else:
            self.hdfs_host = hdfs_host or "localhost"
            self.hdfs_port = hdfs_port or 9000
        self.hdfs_user = hdfs_user
        self.hdfs_url = f"hdfs://{hdfs_host}:{hdfs_port}"
        
        # Estrutura de diretórios no HDFS
        self.hdfs_structure = {
            'raw_data': '/bigdata/finance/raw',
            'processed_data': '/bigdata/finance/processed',
            'events_data': '/bigdata/finance/events',
            'ml_models': '/bigdata/finance/models',
            'checkpoints': '/bigdata/checkpoints',
            'logs': '/bigdata/logs'
        }
    
    def check_hdfs_connection(self) -> bool:
        """
        Verifica conexão com HDFS
        
        Returns:
            True se conectado, False caso contrário
        """
        try:
            result = subprocess.run(
                ['hdfs', 'dfs', '-ls', '/'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Conexão HDFS estabelecida com sucesso")
                return True
            else:
                logger.warning(f"Falha na conexão HDFS: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout na conexão HDFS")
            return False
        except FileNotFoundError:
            logger.error("Comando hdfs não encontrado. Verifique instalação do Hadoop")
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar conexão HDFS: {str(e)}")
            return False
    
    def create_directory_structure(self) -> bool:
        """
        Cria estrutura de diretórios no HDFS
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            for dir_name, dir_path in self.hdfs_structure.items():
                result = subprocess.run(
                    ['hdfs', 'dfs', '-mkdir', '-p', dir_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Diretório criado: {dir_path}")
                else:
                    logger.warning(f"Falha ao criar {dir_path}: {result.stderr}")
            
            # Definir permissões
            for dir_path in self.hdfs_structure.values():
                subprocess.run(
                    ['hdfs', 'dfs', '-chmod', '755', dir_path],
                    capture_output=True
                )
            
            logger.info("Estrutura de diretórios HDFS criada")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar estrutura HDFS: {str(e)}")
            return False
    
    def upload_file(self, local_path: str, hdfs_path: str) -> bool:
        """
        Faz upload de arquivo para HDFS
        
        Args:
            local_path: Caminho local do arquivo
            hdfs_path: Caminho de destino no HDFS
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f"Arquivo local não encontrado: {local_path}")
                return False
            
            result = subprocess.run(
                ['hdfs', 'dfs', '-put', local_path, hdfs_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Arquivo enviado: {local_path} -> {hdfs_path}")
                return True
            else:
                logger.error(f"Falha no upload: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no upload: {str(e)}")
            return False
    
    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """
        Faz download de arquivo do HDFS
        
        Args:
            hdfs_path: Caminho no HDFS
            local_path: Caminho local de destino
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            # Criar diretório local se não existir
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            result = subprocess.run(
                ['hdfs', 'dfs', '-get', hdfs_path, local_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Arquivo baixado: {hdfs_path} -> {local_path}")
                return True
            else:
                logger.error(f"Falha no download: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no download: {str(e)}")
            return False
    
    def list_directory(self, hdfs_path: str) -> List[Dict]:
        """
        Lista conteúdo de diretório HDFS
        
        Args:
            hdfs_path: Caminho do diretório no HDFS
            
        Returns:
            Lista com informações dos arquivos
        """
        try:
            result = subprocess.run(
                ['hdfs', 'dfs', '-ls', hdfs_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Erro ao listar diretório: {result.stderr}")
                return []
            
            files_info = []
            lines = result.stdout.strip().split('\n')[1:]  # Pular cabeçalho
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 8:
                        files_info.append({
                            'permissions': parts[0],
                            'replication': parts[1],
                            'owner': parts[2],
                            'group': parts[3],
                            'size': int(parts[4]),
                            'date': parts[5],
                            'time': parts[6],
                            'path': parts[7]
                        })
            
            return files_info
            
        except Exception as e:
            logger.error(f"Erro ao listar diretório: {str(e)}")
            return []
    
    def delete_file(self, hdfs_path: str) -> bool:
        """
        Remove arquivo ou diretório do HDFS
        
        Args:
            hdfs_path: Caminho no HDFS
            
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            result = subprocess.run(
                ['hdfs', 'dfs', '-rm', '-r', hdfs_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Arquivo removido: {hdfs_path}")
                return True
            else:
                logger.error(f"Falha ao remover: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao remover arquivo: {str(e)}")
            return False
    
    def get_file_info(self, hdfs_path: str) -> Optional[Dict]:
        """
        Obtém informações de um arquivo HDFS
        
        Args:
            hdfs_path: Caminho do arquivo no HDFS
            
        Returns:
            Dicionário com informações do arquivo ou None
        """
        try:
            result = subprocess.run(
                ['hdfs', 'dfs', '-stat', '%n,%o,%r,%b,%y', hdfs_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                info = result.stdout.strip().split(',')
                return {
                    'name': info[0],
                    'owner': info[1],
                    'replication': int(info[2]),
                    'size': int(info[3]),
                    'modification_time': info[4]
                }
            else:
                logger.error(f"Arquivo não encontrado: {hdfs_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao obter info do arquivo: {str(e)}")
            return None
    
    def get_cluster_status(self) -> Dict:
        """
        Obtém status do cluster HDFS
        
        Returns:
            Dicionário com informações do cluster
        """
        try:
            # Status do filesystem
            fs_result = subprocess.run(
                ['hdfs', 'dfsadmin', '-report'],
                capture_output=True,
                text=True
            )
            
            status = {
                'connected': fs_result.returncode == 0,
                'hdfs_url': self.hdfs_url,
                'directories': self.hdfs_structure
            }
            
            if fs_result.returncode == 0:
                # Extrair informações básicas do relatório
                output = fs_result.stdout
                if 'Configured Capacity:' in output:
                    lines = output.split('\n')
                    for line in lines:
                        if 'Configured Capacity:' in line:
                            status['total_capacity'] = line.split(':')[1].strip()
                        elif 'DFS Used:' in line:
                            status['used_space'] = line.split(':')[1].strip()
                        elif 'DFS Remaining:' in line:
                            status['remaining_space'] = line.split(':')[1].strip()
            
            return status
            
        except Exception as e:
            logger.error(f"Erro ao obter status do cluster: {str(e)}")
            return {'connected': False, 'error': str(e)}
    
    def setup_hdfs_environment(self) -> bool:
        """
        Configura ambiente HDFS completo
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            logger.info("Configurando ambiente HDFS...")
            
            # Verificar conexão
            if not self.check_hdfs_connection():
                logger.error("Não foi possível conectar ao HDFS")
                return False
            
            # Criar estrutura de diretórios
            if not self.create_directory_structure():
                logger.error("Falha ao criar estrutura de diretórios")
                return False
            
            logger.info("Ambiente HDFS configurado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro na configuração HDFS: {str(e)}")
            return False

# Exemplo de uso
if __name__ == "__main__":
    hdfs_manager = HDFSManager()
    
    # Verificar status
    status = hdfs_manager.get_cluster_status()
    print(f"Status HDFS: {status}")
    
    # Configurar ambiente
    if hdfs_manager.setup_hdfs_environment():
        print("Ambiente HDFS configurado com sucesso")
    else:
        print("Falha na configuração do HDFS")