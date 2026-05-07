import os
from waitress import serve
from app import app

if __name__ == '__main__':
    # 生产环境使用waitress服务器
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    serve(app, host='0.0.0.0', port=port)