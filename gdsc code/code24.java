import java.util.Random;
import java.util.Scanner;

public class PasswordGenerator {

    public static void main(String[] args) {
        
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the length of the password you want:");
        int length = scanner.nextInt();

       
        String password = generatePassword(length);

        System.out.println("Your password is: " + password);
    }

    private static String generatePassword(int length) {
        
        String characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()";

        
        Random random = new Random();

       
        StringBuffer password = new StringBuffer();
      
        for (int i = 0; i < length; i++) {
            int index = random.nextInt(characters.length());
            char character = characters.charAt(index);
            password.append(character);
        }

        
        return password.toString();
    }
}
