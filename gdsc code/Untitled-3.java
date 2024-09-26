import java.io.*;
import javax.servelet.*;
import javax.servlet.http.*;
public class HelloServlet extends HttpServlet
{
    public void dopost(HttpServletRequest req,HttpServletResponse res)throw servelet
    {
        res.setContentType("text/html");
        printWriter pw=res.getWriter();
        String s=req.getParametr("u");
        pw.println("welcome"+s);
        pw.close();
    }
}