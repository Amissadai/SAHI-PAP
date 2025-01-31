from minio import Minio
from minio.error import S3Error

def configuracao_servidor():
    # Configuração do cliente MinIO
    minio_client = Minio(
        "172.100.11.116:9091",  # URL do MinIO
        access_key="Bv4fweTq8jzUnwefDimR",
        secret_key="yirEms8Sy0q0IOEkswljBrb62YS1WCwrLbqlXv6h",
        secure=False  # True se usar HTTPS, False se HTTP
    )

    bucket_name = "healthtech-paponline"
    return minio_client, bucket_name


def criar_bucket(minio_client, bucket_name):
    # Criar o bucket se ele não existir
    found = minio_client.bucket_exists(bucket_name)
    if not found:
        minio_client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' criado.")
    else:
        print(f"Bucket '{bucket_name}' já existe.")


def enviar_dado_minio(minio_client, bucket_name, arquivo_local, arquivo_destino):
    try:
        minio_client.fput_object(bucket_name, arquivo_destino, arquivo_local)
        print(f"Arquivo '{arquivo_local}' enviado como '{arquivo_destino}' com sucesso.")

        # Gerar um link temporário para acessar o arquivo
        url = minio_client.presigned_get_object(bucket_name, arquivo_destino)
        print(f"Link para acessar o arquivo: {url}")
    except S3Error as e:
        print(f"Erro ao enviar arquivo: {e}")


def download_dado_minio(minio_client, bucket_name, arquivo_remoto, arquivo_local):
    try:
        #url = minio_client.presigned_get_object(bucket_name, arquivo_remoto)
        minio_client.fget_object(bucket_name, arquivo_remoto, arquivo_local)
        print(f"Arquivo '{arquivo_remoto}' baixado com sucesso para '{arquivo_local}'.")
        #print(f"ID do dado: {url}")
    except S3Error as e:
        print(f"Erro ao baixar arquivo: {e}")
        

'''
def main():
    # Nome do arquivo local e nome desejado no MinIO
    arquivo_local = "04927.bmp"
    arquivo_destino = "dataset/04927.bmp"
    arquivo_remoto = "dataset/resultado_final.jpg"  # Certifique-se de que o caminho está correto

    minio_client, bucket_name = configuracao_servidor()
    criar_bucket(minio_client, bucket_name)
    enviar_dado_minio(minio_client, bucket_name, arquivo_local, arquivo_destino)
    #download_dado_minio(minio_client, bucket_name, arquivo_remoto, "downloaded_imagem.jpg")


if __name__ == "__main__":
    main()
'''