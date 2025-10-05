import streamlit as st
import json
import os


CONFIG_FILE = os.path.join(os.path.dirname(__file__), '.aws_config.json')


def save_config_to_file(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                'aws_config': config,
                'aws_configured': True
            }, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save config to file: {e}")
        return False


def load_config_from_file():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return data.get('aws_config'), data.get('aws_configured', False)
    except Exception as e:
        st.warning(f"Could not load saved config: {e}")
    return None, False


def initialize_aws_session_state():
    saved_config, was_configured = load_config_from_file()
    defaults = {
        'aws_configured': was_configured if saved_config else False,
        'aws_config': saved_config if saved_config else {
            's3_bucket': '',
            'region': 'us-east-1',
            'use_textract': True,
            'use_comprehend': True,
            'use_rekognition': False,
            'use_bedrock': False,
            'use_cloudwatch': True
        },
        'aws_credentials_valid': False,
        'aws_documents_processed': False,
        'aws_chat_history': [],
        'aws_tts_audio': {},
        'aws_search_results': None,
        'aws_last_response': None,
        'aws_processing_result': None
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            if isinstance(default_value, dict):
                st.session_state[key] = default_value.copy()
            elif isinstance(default_value, list):
                st.session_state[key] = default_value.copy()
            else:
                st.session_state[key] = default_value
    if saved_config and st.session_state.get('aws_config', {}).get('s3_bucket', '') == '':
        st.session_state['aws_config'] = saved_config.copy()
        st.session_state['aws_configured'] = was_configured


def get_aws_config():
    initialize_aws_session_state()
    config = st.session_state.get('aws_config', None)
    if not config or not isinstance(config, dict) or not config.get('s3_bucket'):
        saved_config, _ = load_config_from_file()
        if saved_config and isinstance(saved_config, dict):
            st.session_state['aws_config'] = saved_config.copy()
            return saved_config.copy()
        default_config = {
            's3_bucket': '',
            'region': 'us-east-1',
            'use_textract': True,
            'use_comprehend': True,
            'use_rekognition': False,
            'use_bedrock': False,
            'use_cloudwatch': True
        }
        return default_config
    return config.copy()


def set_aws_config(config, configured=True):
    st.session_state['aws_config'] = config.copy()
    st.session_state['aws_configured'] = configured
    save_config_to_file(config)


def validate_aws_config():
    if not st.session_state.get('aws_configured', False):
        saved_config, was_configured = load_config_from_file()
        if saved_config and was_configured:
            st.session_state['aws_config'] = saved_config
            st.session_state['aws_configured'] = True
        else:
            return False, "AWS not configured"
    config = st.session_state.get('aws_config', {})
    if not config.get('s3_bucket', '').strip():
        return False, "S3 bucket not specified or empty"
    if not config.get('region'):
        return False, "AWS region not specified"
    return True, "Configuration valid"


def is_valid_input(user_input):
    try:
        from common import is_valid_input as common_is_valid
        return common_is_valid(user_input)
    except ImportError:
        if not user_input or not user_input.strip():
            return False
        if len(user_input) < 3:
            return False
        if not any(char.isalnum() for char in user_input):
            return False
        return True


def check_aws_credentials():
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            try:
                s3_client = boto3.client('s3')
                s3_client.list_buckets()
                st.session_state.aws_credentials_valid = True
                return True, "Credentials valid and S3 accessible"
            except Exception as e:
                st.session_state.aws_credentials_valid = False
                return False, f"Credentials found but S3 access failed: {str(e)}"
        else:
            st.session_state.aws_credentials_valid = False
            return False, "No AWS credentials found"
    except ImportError:
        return False, "boto3 not installed"
    except Exception as e:
        st.session_state.aws_credentials_valid = False
        return False, f"Credential check failed: {str(e)}"


def check_aws_prerequisites(allow_force_enable=False, require_documents=False):
    initialize_aws_session_state()
    try:
        import boto3
    except ImportError:
        st.error("AWS integration not available. Please install boto3: pip install boto3")
        return False
    aws_configured = st.session_state.get('aws_configured', False)
    aws_config = st.session_state.get('aws_config', {})
    if not aws_configured or not aws_config.get('s3_bucket'):
        saved_config, was_configured = load_config_from_file()
        if saved_config and was_configured:
            st.session_state['aws_config'] = saved_config
            st.session_state['aws_configured'] = True
            aws_config = saved_config
            aws_configured = True
            st.info("Loaded saved AWS configuration")
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    st.json(json.load(f))
            except:
                st.error("Could not read config file")
    if not isinstance(aws_config, dict):
        st.warning("AWS configuration is corrupted. Please reconfigure.")
        if st.button("Go to Configuration", key="corrupted_config"):
            st.switch_page("pages/aws_configuration.py")
        return False
    if not aws_configured:
        st.warning("AWS not configured. Please configure AWS services first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to AWS Configuration", key="prereq_config"):
                st.switch_page("pages/aws_configuration.py")
        if allow_force_enable:
            with col2:
                if st.button("Force Configure (Testing)", key="force_config"):
                    if aws_config.get('s3_bucket', '').strip():
                        st.session_state.aws_configured = True
                        st.session_state.aws_config = aws_config
                        save_config_to_file(aws_config)
                        st.success("AWS configuration force-enabled")
                        st.rerun()
                    else:
                        st.error("Cannot force enable without a bucket name. Please configure first.")
        return False
    bucket_name = aws_config.get('s3_bucket', '').strip()
    region = aws_config.get('region', '')
    if not bucket_name or not region:
        st.error("AWS configuration incomplete - missing bucket or region")
        st.write(f"**Current bucket:** '{bucket_name or 'NOT SET'}'")
        st.write(f"**Current region:** '{region or 'NOT SET'}'")
        if st.button("Fix Configuration", key="fix_config"):
            st.switch_page("pages/aws_configuration.py")
        return False
    if require_documents:
        docs_processed = st.session_state.get('aws_documents_processed', False)
        if not docs_processed:
            st.warning("No documents have been processed with AWS yet.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Go to Document Manager", key="prereq_docs"):
                    st.switch_page("pages/aws_document_manager.py")
            if allow_force_enable:
                with col2:
                    if st.button("Skip Check (Testing)", key="skip_docs_check"):
                        st.session_state.aws_documents_processed = True
                        st.success("Document processing check skipped")
                        st.rerun()
            return False
    return True


def render_navigation(current_page=None):
    col1, col2, col3 = st.columns(3)
    with col1:
        if current_page != 'menu':
            if st.button("Back to AWS Menu", key="nav_back_aws"):
                st.switch_page("pages/aws_mode_selection.py")
    with col2:
        if current_page != 'config':
            if st.button("AWS Configuration", key="nav_config"):
                st.switch_page("pages/aws_configuration.py")
    with col3:
        if current_page != 'docs':
            if st.button("Document Manager", key="nav_docs"):
                st.switch_page("pages/aws_document_manager.py")


def initialize_s3_manager(aws_config):
    try:
        bucket_name = aws_config.get('s3_bucket', '').strip()
        region = aws_config.get('region', '')
        if not bucket_name or not region:
            st.error(f"Invalid AWS config - Bucket: '{bucket_name}', Region: '{region}'")
            st.error("Please configure AWS settings with a valid bucket name")
            return None
        try:
            from aws_s3_integration import AWSS3DocumentManager
            s3_manager = AWSS3DocumentManager(bucket_name, region)
            st.success(f"S3 Manager initialized for bucket '{bucket_name}' in region '{region}'")
            return s3_manager
        except ImportError:
            st.error("AWS integration files not found. Please ensure aws_s3_integration.py is available.")
            st.info("You can still use the basic document management features below.")
            return None
    except Exception as e:
        st.error(f"Failed to initialize S3 manager: {str(e)}")
        return None


def create_s3_bucket_if_not_exists(bucket_name, region):
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        s3_client = boto3.client('s3', region_name=region)
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            st.success(f"Bucket '{bucket_name}' already exists and is accessible")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if region == 'us-east-1':
                        s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region}
                        )
                    st.success(f"Created S3 bucket '{bucket_name}' in region '{region}'")
                    return True
                except ClientError as create_error:
                    error_code = create_error.response['Error']['Code']
                    if error_code == 'BucketAlreadyOwnedByYou':
                        st.success(f"Bucket '{bucket_name}' already exists")
                        return True
                    elif error_code == 'BucketAlreadyExists':
                        st.error(f"Bucket name '{bucket_name}' is already taken globally. Try a different name.")
                        return False
                    else:
                        st.error(f"Failed to create bucket: {str(create_error)}")
                        return False
            else:
                st.error(f"Error accessing bucket: {str(e)}")
                return False
    except NoCredentialsError:
        st.warning("AWS credentials not found. Run 'aws configure' to set them up.")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False


def get_region_index(region):
    regions = ['us-east-1', 'us-west-2', 'us-east-2', 'us-west-1',
               'eu-west-1', 'eu-central-1', 'ap-southeast-1', 'ap-northeast-1']
    try:
        return regions.index(region)
    except ValueError:
        return 0


def render_aws_setup_instructions():
    st.subheader("AWS Setup Instructions")
    st.markdown("""
    ### To set up AWS credentials:
    1. **Install AWS CLI**: `pip install awscli`
    2. **Configure credentials**: `aws configure`
    3. **Enter your credentials**:
       - AWS Access Key ID
       - AWS Secret Access Key  
       - Default region name
       - Default output format (json)

    ### Alternative: Environment Variables
    Set these environment variables:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_DEFAULT_REGION`
    """)