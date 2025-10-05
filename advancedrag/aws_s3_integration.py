

import boto3
import os
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AWSS3DocumentManager:
    def __init__(self, bucket_name: str, region_name: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region_name = region_name
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.s3_resource = boto3.resource('s3', region_name=region_name)
            self.bucket = self.s3_resource.Bucket(bucket_name)
            self._ensure_bucket_exists()
            logger.info(f"AWS S3 client initialized for bucket: {bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS S3 client: {str(e)}")
            raise

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    if self.region_name == 'eu-west-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region_name}
                        )
                    logger.info(f"Created bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {str(create_error)}")
                    raise
            else:
                logger.error(f"Error accessing bucket: {str(e)}")
                raise

    def upload_document(self, local_file_path: str, s3_key: str = None) -> str:
        if not s3_key:
            s3_key = f"documents/{os.path.basename(local_file_path)}"
        try:
            metadata = {
                'upload_timestamp': datetime.now().isoformat(),
                'original_filename': os.path.basename(local_file_path),
                'file_size': str(os.path.getsize(local_file_path))
            }
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'Metadata': metadata}
            )
            logger.info(f"Uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {str(e)}")
            raise

    def download_document(self, s3_key: str, local_path: str = None) -> str:
        if not local_path:
            local_path = f"/tmp/{os.path.basename(s3_key)}"
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {str(e)}")
            raise

    def list_documents(self, prefix: str = "documents/") -> List[Dict]:
        documents = []
        try:
            for obj in self.bucket.objects.filter(Prefix=prefix):
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=obj.key)
                doc_info = {
                    'key': obj.key,
                    'filename': os.path.basename(obj.key),
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'content_type': response.get('ContentType', ''),
                    'metadata': response.get('Metadata', {})
                }
                documents.append(doc_info)
            logger.info(f"Found {len(documents)} documents in bucket")
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise

    def delete_document(self, s3_key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {s3_key}: {str(e)}")
            return False

    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {s3_key}: {str(e)}")
            raise

    def document_exists(self, s3_key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def get_document_metadata(self, s3_key: str) -> Dict:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', ''),
                'metadata': response.get('Metadata', {})
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {s3_key}: {str(e)}")
            return {}


def process_files_from_s3(s3_manager: AWSS3DocumentManager, s3_key: str):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(s3_key)[1]) as tmp_file:
        local_path = s3_manager.download_document(s3_key, tmp_file.name)
        try:
            from processing import process_files
            result = process_files(local_path)
            return result
        except ImportError:
            return basic_file_processing(local_path)


def basic_file_processing(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [(f.read(), 1)]
        elif ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    return [(page.extract_text(), i + 1) for i, page in enumerate(pdf_reader.pages)
                            if page.extract_text()]
            except ImportError:
                return [("PDF processing requires PyPDF2", 1)]
        else:
            return [("Unsupported file type", 1)]
    except Exception as e:
        return [(f"Error processing file: {str(e)}", 1)]

