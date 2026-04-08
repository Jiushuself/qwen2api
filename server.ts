// 简单的文件服务器
import { serve } from "https://deno.land/std@0.208.0/http/file_server.ts";

serve({
  port: 3000,
  fsRoot: "./",
  urlRoot: ""
});

console.log("🌐 Web 界面已启动：http://localhost:3000");
console.log("📄 打开浏览器访问：http://localhost:3000/index.html");
