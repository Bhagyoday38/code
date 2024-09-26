import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);


        System.out.println("Enter a number: ");
        int number = scanner.nextInt();

       
        boolean isPrime = true;
        for (int i = 2; i <= number / 2; i++) {
            if (number % i == 0) {
                isPrime = false;
                break;
            }
        }
        if (isPrime)
 {
            System.out.println("The number is prime.");
        } 
else
 {
            System.out.println("The number is not prime.");
        }
    }
}
