import os
import sys
import streamlit as st

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from styling.styles import get_css
from aws_shared_utils import (
    initialize_aws_session_state,
    render_navigation,
    check_aws_credentials,
    create_s3_bucket_if_not_exists,
    get_region_index,
    render_aws_setup_instructions
)


def main():
    st.set_page_config(page_title="AWS Configuration", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    initialize_aws_session_state()
    st.markdown('<h1 class="main-header">AWS Configuration</h1>', unsafe_allow_html=True)
    render_navigation(current_page='config')
    st.markdown("---")
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        render_credentials_status()
        render_configuration_form()
    except ImportError:
        handle_missing_boto3()


def render_credentials_status():
    st.subheader("AWS Credentials Status")
    credentials_valid, message = check_aws_credentials()
    if credentials_valid:
        st.success(f" {message}")
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_client.list_buckets()
            st.success("AWS S3 access confirmed!")
        except Exception as e:
            st.warning(f"AWS S3 access issue: {str(e)}")
    else:
        st.warning(f"{message}")
        st.info("You can still configure StorySage. AWS credentials can be set up later using 'aws configure'")
    if st.session_state.aws_configured:
        st.success(" AWS Services Configuration Saved")
        with st.expander("Current Configuration", expanded=False):
            st.json(st.session_state.aws_config)
    else:
        st.info(" AWS services not yet configured")


def render_configuration_form():
    st.subheader("AWS Services Configuration")
    current_config = st.session_state.aws_config
    with st.form("aws_services_config"):
        st.markdown("#### Required Services")
        s3_bucket = st.text_input(
            "S3 Bucket Name",
            value=current_config.get('s3_bucket', ''),
            placeholder="Enter your S3 bucket name (e.g., my-documents-bucket)",
            help="Enter a unique bucket name. It must be globally unique across all AWS accounts."
        )
       
        aws_region = st.selectbox(
            "AWS Region",
            options=['us-east-1', 'us-west-2', 'us-east-2', 'us-west-1',
                     'eu-west-1', 'eu-central-1', 'ap-southeast-1', 'ap-northeast-1'],
            index=get_region_index(current_config.get('region', 'us-east-1'))
        )
        st.markdown("#### Optional Services")
        col1, col2 = st.columns(2)
        with col1:
            use_textract = st.checkbox(
                "Enable AWS Textract",
                value=current_config.get('use_textract', True),
                help="Better PDF text extraction"
            )
            use_comprehend = st.checkbox(
                "Enable AWS Comprehend",
                value=current_config.get('use_comprehend', True),
                help="Text analysis and entity recognition"
            )
            use_rekognition = st.checkbox(
                "Enable AWS Rekognition",
                value=current_config.get('use_rekognition', False),
                help="OCR for images"
            )
        with col2:
            use_bedrock = st.checkbox(
                "Enable AWS Bedrock",
                value=current_config.get('use_bedrock', False),
                help="Enterprise LLM access"
            )
            use_cloudwatch = st.checkbox(
                "Enable CloudWatch Logging",
                value=current_config.get('use_cloudwatch', True),
                help="Centralized logging"
            )
        if st.form_submit_button("Save Configuration", type="primary"):
            save_aws_configuration(s3_bucket, aws_region, use_textract, use_comprehend,
                                   use_rekognition, use_bedrock, use_cloudwatch)


def validate_bucket_name(bucket_name):
    import re
    if not bucket_name:
        return False, "Bucket name cannot be empty"
    if len(bucket_name) < 3 or len(bucket_name) > 63:
        return False, "Bucket name must be between 3 and 63 characters"
    if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$', bucket_name):
        return False, "Bucket name must start and end with a letter/number, and contain only lowercase letters, numbers, and hyphens"
    if '..' in bucket_name or '.-' in bucket_name or '-.' in bucket_name:
        return False, "Bucket name cannot contain consecutive periods or period-hyphen combinations"
    return True, "Valid bucket name"


def save_aws_configuration(s3_bucket, aws_region, use_textract, use_comprehend,
                           use_rekognition, use_bedrock, use_cloudwatch):
    s3_bucket = s3_bucket.strip().lower()
    is_valid, message = validate_bucket_name(s3_bucket)
    if not is_valid:
        st.error(f"Invalid bucket name: {message}")
        return
    aws_config = {
        's3_bucket': s3_bucket,
        'region': aws_region,
        'use_textract': use_textract,
        'use_comprehend': use_comprehend,
        'use_rekognition': use_rekognition,
        'use_bedrock': use_bedrock,
        'use_cloudwatch': use_cloudwatch
    }
    if not create_s3_bucket_if_not_exists(s3_bucket, aws_region):
        st.error("Failed to create/access bucket. Please check the bucket name and your AWS permissions.")
        return
    st.session_state['aws_config'] = aws_config
    st.session_state['aws_configured'] = True
    st.session_state.update({
        'aws_config': aws_config,
        'aws_configured': True
    })
    st.success(" AWS configuration saved successfully!")
    st.info(f"S3 Bucket: {s3_bucket}")
    st.info(f" Region: {aws_region}")
    st.info(" You can now use the Document Manager and AWS Chat Mode.")
    with st.expander("Saved Configuration", expanded=True):
        st.json(aws_config)
    st.success(f"Configuration verified: Bucket '{st.session_state.aws_config.get('s3_bucket')}' is set")
    st.info("Click 'Document Manager' in the navigation to proceed")


def handle_missing_boto3():
    st.subheader("AWS Services Configuration")
    st.info(" You can configure settings now and install boto3 later.")
    current_config = st.session_state.aws_config
    with st.form("aws_services_config_no_boto3"):
        s3_bucket = st.text_input(
            "S3 Bucket Name",
            value=current_config.get('s3_bucket', ''),
            placeholder="Enter your S3 bucket name",
            help="Enter a unique bucket name following AWS naming rules"
        )
        aws_region = st.selectbox(
            "AWS Region",
            options=['us-east-1', 'us-west-2', 'us-east-2', 'us-west-1',
                     'eu-west-1', 'eu-central-1', 'ap-southeast-1', 'ap-northeast-1'],
            index=get_region_index(current_config.get('region', 'us-east-1'))
        )
        if st.form_submit_button("Save Basic Configuration"):
            s3_bucket = s3_bucket.strip().lower()
            is_valid, message = validate_bucket_name(s3_bucket)
            if not is_valid:
                st.error(f"Invalid bucket name: {message}")
                return
            aws_config = {
                's3_bucket': s3_bucket,
                'region': aws_region,
                'use_textract': True,
                'use_comprehend': True,
                'use_rekognition': False,
                'use_bedrock': False,
                'use_cloudwatch': True
            }
            st.session_state['aws_config'] = aws_config
            st.session_state['aws_configured'] = True
            st.success("Basic AWS configuration saved!")
            st.info(f" S3 Bucket: {s3_bucket}")
    render_aws_setup_instructions()


if __name__ == "__main__":
    main()