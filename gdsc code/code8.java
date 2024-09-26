import java.util.Scanner;

public class Palindrome 
{
    public static void main(String[] args) 
{
        Scanner scanner = new Scanner(System.in);

        System.out.println("Enter a number: ");
        int number = scanner.nextInt();

        int reversedNumber = 0;
        int originalNumber = number;
        while (number > 0)
 {
            reversedNumber = reversedNumber * 10 + number % 10;
            number /= 10;
        }

        if (originalNumber == reversedNumber) 
{
            System.out.println("The number is a palindrome.");
        } 
else 
{
            System.out.println("The number is not a palindrome.");
        }
    }
}
