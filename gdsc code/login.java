import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
public class login extends HttpServlet
{
    public void dopost(HttpServletRequest req, HttpServletResponse res)throws servelet
    {
        res.setContenttype("text/html");
        printWriter pw = res.getWriter();
        String s=req.getParametr("p");
        if(s.equals("bhagyoday"))
        {
            RequestDispatcher rd=req.getRequestDispatcher("welcomeservlet");
            rd.forward(req,res);
        }
        else
        {
            pw.println("sorry username or password error !!!");
        }
        pw.close();

    }
}